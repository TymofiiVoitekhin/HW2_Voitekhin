from langchain_core.tools import tool
from pydantic import BaseModel, Field

# ── 1. Звичайні інструменти (з попереднього ДЗ) ───────────────────

class InventoryInput(BaseModel):
    component_name: str = Field(description='Назва електронного компонента (наприклад, "MaUWB_ESP32S3", "RELAY_5V")')

@tool(args_schema=InventoryInput)
def check_inventory(component_name: str) -> str:
    """Перевіряє наявність компонента на складі та його характеристики."""
    db = {
        'MAUWB_ESP32S3': {'stock': 10, 'voltage': '3.3V'},
        'RELAY_5V': {'stock': 50, 'voltage': '5.0V'},
        'BME280': {'stock': 24, 'voltage': '3.3V'}
    }
    item = db.get(component_name.upper())
    if item:
        return f"[{component_name}] В наявності: {item['stock']} шт. Напруга: {item['voltage']}."
    return f"[{component_name}] Не знайдено на складі."

class CompatibilityInput(BaseModel):
    device_a: str = Field(description='Перший пристрій')
    device_b: str = Field(description='Другий пристрій')

@tool(args_schema=CompatibilityInput)
def check_compatibility(device_a: str, device_b: str) -> str:
    """Аналізує сумісність логічних рівнів двох пристроїв."""
    if "5V" in device_a or "5V" in device_b:
        return f"УВАГА: Потрібен конвертер логічних рівнів між {device_a} та {device_b}."
    return f"{device_a} та {device_b} сумісні по напрузі."

# ── 2. Ризикові інструменти (для майбутнього HITL) ────────────────

class OrderInput(BaseModel):
    component_name: str = Field(description='Назва компонента для замовлення')
    quantity: int = Field(description='Кількість одиниць')

@tool(args_schema=OrderInput)
def order_components(component_name: str, quantity: int) -> str:
    """Замовити партію компонентів у постачальника (РИЗИКОВА ДІЯ)."""
    # В майбутньому тут буде зупинка графа (interrupt)
    return f"Успішно замовлено {quantity} шт. компонента {component_name}."

class FlashInput(BaseModel):
    board_target: str = Field(description='Назва плати (наприклад, MaUWB_ESP32S3)')
    firmware_version: str = Field(description='Версія прошивки (наприклад, v1.2.0)')

@tool(args_schema=FlashInput)
def flash_firmware(board_target: str, firmware_version: str) -> str:
    """Запустити процес прошивки мікроконтролера (РИЗИКОВА ДІЯ)."""
    # В майбутньому тут буде зупинка графа (interrupt)
    return f"Плату {board_target} успішно прошито версією {firmware_version}."

# Експортуємо всі інструменти для використання у графі
tools_list = [check_inventory, check_compatibility, order_components, flash_firmware]
tools_by_name = {t.name: t for t in tools_list}