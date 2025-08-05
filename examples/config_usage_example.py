#!/usr/bin/env python3
"""
Configuration Usage Examples
Демонстрація використання нової YAML-based конфігураційної системи.
"""

import sys
from pathlib import Path

# Add parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config_manager import get_config_manager

def example_basic_usage():
    """Базове використання ConfigManager"""
    print("🔧 Приклад 1: Базове використання ConfigManager")
    print("="*50)
    
    # Отримати глобальний ConfigManager
    config = get_config_manager()
    
    # Отримати default налаштування pipeline
    defaults = config.get_default_pipeline_config()
    print(f"📦 Default SDXL модель: {defaults.sdxl_model}")
    print(f"👁️ Default face detection: {defaults.face_detection_model}")
    print(f"🎭 Default тип персонажа: {defaults.character_type}")
    print(f"⚡ Default рівень покращення: {defaults.enhancement_level}")
    print(f"🎨 Default якість: {defaults.quality_preset}")
    print()

def example_model_presets():
    """Робота з model presets"""
    print("🎯 Приклад 2: Model Presets")
    print("="*30)
    
    config = get_config_manager()
    
    # Отримати моделі для різних presets
    presets = ['basic', 'advanced', 'professional']
    
    for preset in presets:
        models = config.get_model_preset_models(preset)
        print(f"{preset.upper():>12}: {', '.join(models)}")
    print()

def example_available_models():
    """Список доступних моделей"""
    print("📦 Приклад 3: Доступні моделі")
    print("="*35)
    
    config = get_config_manager()
    
    # SDXL моделі
    sdxl_models = config.get_available_models()
    print("🎨 SDXL моделі:")
    for model in sdxl_models:
        model_info = config.get_model_info(model)
        speciality = model_info.get('speciality', 'general')
        print(f"  • {model} - {speciality}")
    
    print()
    
    # Face detection моделі
    face_models = config.get_available_face_models()
    print("👁️ Face detection моделі:")
    for model in face_models:
        print(f"  • {model}")
    print()

def example_enhancement_settings():
    """Налаштування рівнів покращення"""
    print("⚡ Приклад 4: Рівні покращення")
    print("="*35)
    
    config = get_config_manager()
    
    levels = ['light', 'medium', 'strong', 'extreme']
    
    for level in levels:
        settings = config.get_enhancement_level_settings(level)
        print(f"{level.upper():>8}: denoising={settings.get('denoising_strength', 'N/A')}, "
              f"cfg={settings.get('cfg_scale', 'N/A')}, "
              f"steps={settings.get('steps', 'N/A')}")
    print()

def example_character_prompts():
    """Character-specific prompts"""
    print("🎭 Приклад 5: Character Prompts")
    print("="*35)
    
    config = get_config_manager()
    
    characters = ['female_portrait', 'male_portrait', 'child_portrait']
    
    for character in characters:
        prompts = config.get_character_prompts(character)
        positive = prompts.get('positive', '')[:80] + '...' if len(prompts.get('positive', '')) > 80 else prompts.get('positive', '')
        print(f"{character:>15}: {positive}")
    print()

def example_adetailer_settings():
    """ADetailer quality presets"""
    print("🎨 Приклад 6: ADetailer Quality Presets")
    print("="*40)
    
    config = get_config_manager()
    
    presets = ['conservative', 'balanced', 'aggressive']
    
    for preset in presets:
        settings = config.get_adetailer_settings(preset)
        print(f"{preset.upper():>12}: confidence={settings.get('confidence', 'N/A')}, "
              f"denoising={settings.get('denoising_strength', 'N/A')}, "
              f"padding={settings.get('mask_padding', 'N/A')}")
    print()

def example_webui_config():
    """WebUI налаштування"""
    print("🌐 Приклад 7: WebUI Configuration")
    print("="*35)
    
    config = get_config_manager()
    
    webui_config = config.get_webui_config()
    print(f"🔗 URL: {config.get_webui_url()}")
    print(f"📍 Port: {webui_config.port}")
    print(f"⏱️ Timeout: {webui_config.api_timeout}s")
    print(f"🔄 Max retries: {webui_config.max_retries}")
    print(f"🎚️ Default sampler: {webui_config.sampler}")
    print(f"⚙️ Default CFG: {webui_config.cfg_scale}")
    print(f"👣 Default steps: {webui_config.steps}")
    print()

def example_config_validation():
    """Валідація конфігурації"""
    print("✅ Приклад 8: Валідація конфігурації")
    print("="*40)
    
    config = get_config_manager()
    
    issues = config.validate_config()
    
    if issues:
        print("⚠️ Знайдені проблеми в конфігурації:")
        for issue in issues:
            print(f"  • {issue}")
    else:
        print("✅ Конфігурація валідна - всі файли знайдені!")
    print()

def example_environment_overrides():
    """Environment variable overrides"""
    print("🌍 Приклад 9: Environment Overrides")
    print("="*40)
    
    config = get_config_manager()
    
    # Приклад отримання налаштувань з environment
    webui_url = config.get_env_setting('WEBUI_API_URL', 'default_url')
    api_key = config.get_env_setting('RUNPOD_API_KEY', 'not_set')
    
    print(f"🔗 WebUI URL (env): {webui_url}")
    print(f"🔑 API Key set: {'Yes' if api_key != 'not_set' else 'No'}")
    print()

def example_recommended_face_model():
    """Рекомендовані face models для різних випадків"""
    print("👁️ Приклад 10: Рекомендовані Face Models")
    print("="*45)
    
    config = get_config_manager()
    
    use_cases = ['preview', 'production', 'professional', 'critical_work']
    
    for use_case in use_cases:
        recommended = config.get_recommended_face_model(use_case)
        print(f"{use_case:>15}: {recommended}")
    print()

def main():
    """Запустити всі приклади"""
    print("🎯 Configuration Manager - Приклади використання")
    print("="*60)
    print()
    
    examples = [
        example_basic_usage,
        example_model_presets,
        example_available_models,
        example_enhancement_settings,
        example_character_prompts,
        example_adetailer_settings,
        example_webui_config,
        example_config_validation,
        example_environment_overrides,
        example_recommended_face_model
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"❌ Помилка в прикладі {i}: {e}")
            print()
    
    print("🎉 Всі приклади завершені!")
    print()
    print("💡 Для зміни defaults - редагуйте config/pipeline_settings.yaml")
    print("🔧 Для зміни моделей - редагуйте config/models.yaml")
    print("🎨 Для зміни промптів - редагуйте config/prompt_settings.yaml")

if __name__ == "__main__":
    main()