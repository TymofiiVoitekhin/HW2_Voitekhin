import operator
from typing import Annotated, TypedDict, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from langgraph.types import interrupt, Command

import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

# Імпортуємо наші інструменти
from tools import tools_list, tools_by_name
from knowledge import search_knowledge

all_tools = tools_list + [search_knowledge]
all_tools_by_name = {t.name: t for t in all_tools}

load_dotenv()

# ============================================================================
# 1. Structured Output Models (Pydantic)
# ============================================================================
class Plan(BaseModel):
    """План виконання задачі."""
    goal: str = Field(description='Головна ціль задачі')
    steps: list[str] = Field(description='Список кроків для досягнення цілі')

class ReplanDecision(BaseModel):
    """Рішення replanner: продовжити, перепланувати або завершити."""
    action: Literal['continue', 'replan', 'finish'] = Field(
        description='continue=виконати наступний крок, replan=змінити план, finish=завершити'
    )
    updated_steps: list[str] | None = Field(
        default=None,
        description='Оновлені кроки (тільки якщо action=replan)',
    )
    reasoning: str = Field(description='Пояснення рішення')

# ============================================================================
# 2. State 
# ============================================================================
class PlanExecuteState(TypedDict):
    messages: Annotated[list, operator.add]
    plan: list[str]           # список поточних кроків
    current_step: int         # індекс поточного кроку
    results: list[str]        # результати виконання кроків
    completed: bool           # прапорець завершення

# ============================================================================
# 3. LLM Setup
# ============================================================================
llm = ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite', temperature=0.1)

# Спеціальні версії LLM, які гарантовано повертають Pydantic-схеми
planner_llm = llm.with_structured_output(Plan)
replanner_llm = llm.with_structured_output(ReplanDecision)

# ============================================================================
# 4. Nodes
# ============================================================================
def planner_node(state: PlanExecuteState) -> dict:
    """Вузол 1: Генерує початковий план."""
    user_msg = state['messages'][0].content if state['messages'] else ''
    prompt = (
        f'Створи план для виконання задачі: {user_msg}\n'
        f'Розбий на 2-5 конкретних кроків. Кожен крок має бути чітким.'
        f'Враховуй, що ти маєш інструменти для: перевірки складу, сумісності, замовлення деталей та прошивки.'
    )
    
    plan = planner_llm.invoke(prompt)
    
    return {
        'plan': plan.steps,
        'current_step': 0,
        'results': [],
        'messages': [AIMessage(content=f'План створено:\n- ' + '\n- '.join(plan.steps))],
        'completed': False
    }

RISKY_TOOLS = {'order_components', 'flash_firmware'}

def executor_node(state: PlanExecuteState) -> dict:
    """Вузол 2: Виконує ОДИН крок плану через інструменти з перевіркою ризиків (HITL)."""
    step_idx = state['current_step']
    plan = state['plan']
    
    if step_idx >= len(plan):
        return {'completed': True}
        
    current_step = plan[step_idx]
    
    llm_with_tools = llm.bind_tools(all_tools)
    response = llm_with_tools.invoke(
        f'Виконай цей крок: "{current_step}"\n'
        f'Попередні результати: {state["results"]}'
    )
    
    if hasattr(response, 'tool_calls') and response.tool_calls:
        # Для простоти беремо перший викликаний інструмент
        tc = response.tool_calls[0]
        tool_name = tc['name']
        tool_args = tc['args']
        
        # ── INTERRUPT: Перевірка на ризиковість ──
        if tool_name in RISKY_TOOLS:
            approval = interrupt({
                'action': tool_name,
                'args': tool_args,
                'message': (
                    f'\n[ЗУПИНКА ГРАФА] Потрібне підтвердження людини!\n'
                    f'Інструмент: {tool_name}\n'
                    f'Параметри: {tool_args}'
                )
            })
            
            # ── RESUME: Обробка рішення людини ──
            if isinstance(approval, dict) and approval.get('approved'):
                # Сценарій Approve (або Edit, якщо змінили параметри)
                final_args = approval.get('edited_args', tool_args)
                tool_fn = all_tools_by_name.get(tool_name)
                tool_result = tool_fn.invoke(final_args)
                result_str = f"{tool_name} (СХВАЛЕНО) -> {tool_result}"
            else:
                # Сценарій Reject
                reason = approval.get("reason", "Відхилено оператором") if isinstance(approval, dict) else "Відхилено"
                result_str = f"{tool_name} -> СКАСОВАНО. Причина: {reason}"
                
            return {
                'current_step': step_idx + 1,
                'results': [*state['results'], f"Крок {step_idx+1} ({current_step}): {result_str}"],
                'messages': [AIMessage(content=f"Крок {step_idx+1}: {result_str}")]
            }
        else:
            # Виконання безпечних інструментів (без зупинки)
            tool_fn = all_tools_by_name.get(tool_name)
            if tool_fn:
                tool_result = tool_fn.invoke(tool_args)
                result_str = f"{tool_name} -> {tool_result}"
                return {
                    'current_step': step_idx + 1,
                    'results': [*state['results'], f"Крок {step_idx+1} ({current_step}): {result_str}"],
                    'messages': [AIMessage(content=f"Крок {step_idx+1}: {result_str}")]
                }

    # Очищення сирого формату, якщо інструмент не викликався
    if isinstance(response.content, list) and len(response.content) > 0:
        result = response.content[0].get('text', str(response.content))
    else:
        result = response.content
        
    return {
        'current_step': step_idx + 1,
        'results': [*state['results'], f"Крок {step_idx+1} ({current_step}): {result}"],
        'messages': [AIMessage(content=f"Виконано крок {step_idx+1}: {result}")]
    }

def replanner_node(state: PlanExecuteState) -> dict:
    """Вузол 3: Аналізує прогрес і приймає рішення."""
    plan = state['plan']
    step_idx = state['current_step']
    results = state['results']
    
    if step_idx >= len(plan):
        return {'completed': True}
        
    remaining = plan[step_idx:]
    prompt = (
        f'Оціни прогрес виконання:\n'
        f'План: {plan}\n'
        f'Виконано кроків: {step_idx}/{len(plan)}\n'
        f'Результати: {results}\n'
        f'Залишилось виконати: {remaining}\n'
        f'Прийми рішення: continue (наступний крок), replan (змінити план), finish (завершити задачу).\n'
        f'УВАГА: Якщо ризикову дію було СКАСОВАНО оператором, або якщо для вирішення проблеми '
        f'потрібні інструменти, яких у тебе немає (наприклад, зв\'язок з людьми) — негайно обирай action="finish" '
        f'та описуй причину завершення.'
    )
    
    decision = replanner_llm.invoke(prompt)
    
    if decision.action == 'finish':
        return {
            'completed': True, 
            'messages': [AIMessage(content=f'Задачу завершено. Висновок: {decision.reasoning}')]
        }
    elif decision.action == 'replan' and decision.updated_steps:
        return {
            'plan': decision.updated_steps, 
            'current_step': 0,
            'messages': [AIMessage(content=f'План змінено. Причина: {decision.reasoning}')]
        }
        
    # Якщо continue — нічого не змінюємо у стані, граф просто піде на наступну ітерацію
    return {}

# ============================================================================
# 5. Маршрутизація та Збірка Графа
# ============================================================================
def should_end(state: PlanExecuteState) -> Literal['executor', '__end__']:
    if state.get('completed'):
        return '__end__'
    return 'executor'

graph = StateGraph(PlanExecuteState)
graph.add_node('planner', planner_node)
graph.add_node('executor', executor_node)
graph.add_node('replanner', replanner_node)

graph.add_edge(START, 'planner')
graph.add_edge('planner', 'executor')
graph.add_edge('executor', 'replanner')
graph.add_conditional_edges('replanner', should_end)

# --- ПІДКЛЮЧЕННЯ БАЗИ ДАНИХ (ЗАВДАННЯ 2) ---
conn = sqlite3.connect('agent_state.db', check_same_thread=False)
saver = SqliteSaver(conn)

# Компілюємо граф з checkpointer
app = graph.compile(checkpointer=saver)

# ============================================================================
# 6. Тестовий запуск (Автоматизована демонстрація двох сценаріїв)
# ============================================================================
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    from langgraph.types import Command
    
    print("\n" + "="*70)
    print("=== ТЕСТ ЗАВДАННЯ 4: HUMAN-IN-THE-LOOP (АВТОМАТИЗОВАНА ДЕМОНСТРАЦІЯ) ===")
    print("="*70 + "\n")
    
    initial_state = {
        'messages': [HumanMessage(content="Дізнайся з бази даних умови постачання MaUWB_ESP32S3, а потім оформи замовлення на 10 таких плат. Якщо замовлення пройде, проший одну плату версією v3.0.")],
        'plan': [],
        'current_step': 0,
        'results': [],
        'completed': False
    }

    # Допоміжна функція для гарного виводу логів
    def process_stream(stream):
        for event in stream:
            for node_name, node_data in event.items():
                if node_data and 'messages' in node_data and node_data['messages']:
                    print(f"[{node_name.upper()}]: {node_data['messages'][-1].content}\n")
                elif node_name == 'replanner':
                    # Додаємо перевірку: якщо є прапорець completed, значить це фініш
                    if node_data and node_data.get('completed'):
                        print(f"[{node_name.upper()}]: Всі кроки плану вичерпано. Роботу завершено.\n")
                    else:
                        print(f"[{node_name.upper()}]: Все йде за планом. Продовжуємо виконання кроків (continue).\n")

    # ---------------------------------------------------------
    # СЦЕНАРІЙ 1: REJECT (Відхилення)
    # ---------------------------------------------------------
    print("\n" + "-"*70)
    print(">>> СЦЕНАРІЙ 1: REJECT (ВІДМОВА ОПЕРАТОРА) <<<")
    print("Мета: Показати, як агент адекватно реагує на заборону ризикової дії.")
    print("-"*70 + "\n")
    
    config_reject = {'configurable': {'thread_id': 'hitl-reject-test-02'}}
    
    print("[СИСТЕМА]: Запуск агента...\n")
    process_stream(app.stream(initial_state, config=config_reject, stream_mode="updates"))
        
    state = app.get_state(config_reject)
    if state.tasks and state.tasks[0].interrupts:
        print("[СИСТЕМА]: Граф призупинено! Очікування підтвердження...")
        print(f"[ДЕТАЛІ ЗУПИНКИ]: {state.tasks[0].interrupts[0].value['message']}\n")
        
        print("[ОПЕРАТОР]: Відхиляю замовлення. Причина: Бюджет компанії перевищено.\n")
        
        print("[СИСТЕМА]: Відновлення роботи агента з рішенням оператора...\n")
        process_stream(app.stream(Command(resume={'approved': False, 'reason': 'Бюджет перевищено, скасовуємо.'}), config=config_reject, stream_mode="updates"))


    # ---------------------------------------------------------
    # СЦЕНАРІЙ 2: APPROVE (Підтвердження)
    # ---------------------------------------------------------
    print("\n" + "-"*70)
    print(">>> СЦЕНАРІЙ 2: APPROVE (ПІДТВЕРДЖЕННЯ ОПЕРАТОРА) <<<")
    print("Мета: Показати повний цикл із успішним підтвердженням замовлення та подальшою прошивкою.")
    print("-"*70 + "\n")
    
    config_approve = {'configurable': {'thread_id': 'hitl-approve-test-02'}}
    
    print("[СИСТЕМА]: Запуск агента (чиста сесія)...\n")
    process_stream(app.stream(initial_state, config=config_approve, stream_mode="updates"))
        
    state = app.get_state(config_approve)
    if state.tasks and state.tasks[0].interrupts:
        print("[СИСТЕМА]: Граф призупинено! Очікування підтвердження...")
        print(f"[ДЕТАЛІ ЗУПИНКИ]: {state.tasks[0].interrupts[0].value['message']}\n")
        
        print("[ОПЕРАТОР]: Підтверджую замовлення. Можна продовжувати.\n")
        
        print("[СИСТЕМА]: Відновлення роботи агента з рішенням оператора...\n")
        process_stream(app.stream(Command(resume={'approved': True}), config=config_approve, stream_mode="updates"))
        
        # Оскільки прошивка (flash_firmware) — це ТАКОЖ ризиковий інструмент, 
        # агент зупиниться ще раз! Давайте підтвердимо і його, щоб план дійшов до кінця.
        state2 = app.get_state(config_approve)
        if state2.tasks and state2.tasks[0].interrupts:
            print("[СИСТЕМА]: Граф знову призупинено (нова ризикова дія)!")
            print(f"[ДЕТАЛІ ЗУПИНКИ]: {state2.tasks[0].interrupts[0].value['message']}\n")
            print("[ОПЕРАТОР]: Підтверджую прошивку плати.\n")
            
            print("[СИСТЕМА]: Відновлення роботи агента...\n")
            process_stream(app.stream(Command(resume={'approved': True}), config=config_approve, stream_mode="updates"))