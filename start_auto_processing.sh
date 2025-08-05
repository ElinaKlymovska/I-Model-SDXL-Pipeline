#!/bin/bash
# Швидкий запуск автоматичної обробки фотографій

echo "🎨 I, Model - Автоматична обробка фотографій"
echo "============================================="

# Перевірити чи в правильній директорії
if [ ! -f "scripts/auto_batch_processor.py" ]; then
    echo "❌ Запустіть з директорії I-Model-SDXL-Pipeline"
    echo "cd /workspace/I-Model-SDXL-Pipeline"
    exit 1
fi

# Створити необхідні директорії
mkdir -p assets/input assets/output assets/temp assets/downloads

echo "📁 Директорії створено"

# Запитати режим роботи
echo ""
echo "Оберіть режим роботи:"
echo "1. 🌐 Web API (браузер інтерфейс)"
echo "2. 💻 CLI Batch (термінал)"
echo "3. 📋 Показати приклади використання"

read -p "Ваш вибір (1-3): " choice

case $choice in
    1)
        echo "🌐 Запуск Web API на порту 5000..."
        echo "🔗 Відкрийте в браузері:"
        echo "   Local: http://localhost:5000"
        if [ ! -z "$RUNPOD_POD_ID" ]; then
            echo "   RunPod: https://$RUNPOD_POD_ID-5000.proxy.runpod.net"
        fi
        echo ""
        python scripts/web_api.py
        ;;
    2)
        echo "💻 CLI Batch режим"
        echo ""
        echo "Приклади команд:"
        echo "  python scripts/auto_batch_processor.py assets/input --preset professional_headshot --package"
        echo "  python scripts/auto_batch_processor.py photo.jpg --model epicrealism_xl --enhancement strong"
        echo ""
        read -p "Введіть команду або Enter для обробки assets/input: " cmd
        
        if [ -z "$cmd" ]; then
            python scripts/auto_batch_processor.py assets/input --preset professional_headshot --package
        else
            eval "$cmd"
        fi
        ;;
    3)
        echo "📋 Приклади використання:"
        echo ""
        echo "🔥 Швидкі команди:"
        echo "  # Професійні портрети"
        echo "  python scripts/auto_batch_processor.py assets/input --preset professional_headshot --package"
        echo ""
        echo "  # Артистичні портрети"
        echo "  python scripts/auto_batch_processor.py assets/input --preset artistic_portrait --model juggernaut_xl_v9 --package"
        echo ""
        echo "  # Швидка обробка (Lightning модель)"
        echo "  python scripts/auto_batch_processor.py assets/input --preset natural_candid --model realvisxl_v5_lightning --package"
        echo ""
        echo "🌐 Web API:"
        echo "  python scripts/web_api.py"
        echo "  # Відкрити http://localhost:5000"
        echo ""
        echo "📊 Доступні presets:"
        echo "  - professional_headshot"
        echo "  - artistic_portrait" 
        echo "  - natural_candid"
        echo "  - glamour_portrait"
        echo ""
        echo "🎯 Доступні моделі:"
        echo "  - epicrealism_xl (рекомендовано)"
        echo "  - realvisxl_v5_lightning (швидкий)"
        echo "  - juggernaut_xl_v9 (деталізований)"
        echo ""
        echo "⚡ Рівні покращення:"
        echo "  - light (мінімальний)"
        echo "  - medium (збалансований)"
        echo "  - strong (сильний)"
        echo "  - extreme (максимальний)"
        ;;
    *)
        echo "❌ Неправильний вибір"
        exit 1
        ;;
esac