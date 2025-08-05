#!/bin/bash
# Швидкий запуск системи корекції лиця

echo "🎭 I, Model - Enhanced Face Correction Pipeline"
echo "==============================================="

# Перевірити чи в правильній директорії
if [ ! -f "scripts/face_correction_pipeline.py" ]; then
    echo "❌ Запустіть з директорії I-Model (face correction pipeline)"
    if [ -d "/workspace/I-Model" ]; then
        echo "cd /workspace/I-Model"
    fi
    exit 1
fi

# Створити необхідні директорії
mkdir -p assets/input assets/output assets/temp assets/downloads

echo "📁 Директорії створено"

# Запитати режим роботи
echo ""
echo "Оберіть режим роботи:"
echo "1. 🎭 Face Correction Demo (демонстрація корекції лиця)"
echo "2. 💻 Single Image Processing (одне зображення)"
echo "3. 📁 Batch Processing (пакетна обробка)"
echo "4. 🌐 Stable Diffusion WebUI (повний інтерфейс)"
echo "5. 📋 Показати приклади використання"

read -p "Ваш вибір (1-5): " choice

case $choice in
    1)
        echo "🎭 Запуск демонстрації корекції лиця..."
        echo ""
        python scripts/demo_face_correction.py
        ;;
    2)
        echo "💻 Обробка одного зображення"
        echo ""
        read -p "Шлях до зображення (або Enter для data/input/перше_зображення): " input_path
        read -p "Директорія результатів (або Enter для ./results/): " output_dir
        
        if [ -z "$input_path" ]; then
            # Find first image in data/input
            input_path=$(find data/input -name "*.jpg" -o -name "*.png" -o -name "*.webp" | head -1)
            if [ -z "$input_path" ]; then
                echo "❌ Зображення не знайдено в data/input"
                exit 1
            fi
        fi
        
        if [ -z "$output_dir" ]; then
            output_dir="./results/"
        fi
        
        echo "🎨 Обробка: $input_path -> $output_dir"
        python scripts/demo_pipeline.py --input "$input_path" --output "$output_dir" \
            --character female_portrait --model copax_realistic_xl --enhancement medium
        ;;
    3)
        echo "📁 Пакетна обробка"
        echo ""
        read -p "Директорія з зображеннями (або Enter для data/input): " input_dir
        read -p "Директорія результатів (або Enter для ./results/): " output_dir
        
        if [ -z "$input_dir" ]; then
            input_dir="data/input"
        fi
        
        if [ -z "$output_dir" ]; then
            output_dir="./results/"
        fi
        
        echo "📁 Пакетна обробка: $input_dir -> $output_dir"
        python scripts/demo_pipeline.py --batch "$input_dir" --output "$output_dir" \
            --character female_portrait --model copax_realistic_xl --enhancement medium
        ;;
    4)
        echo "🌐 Запуск Stable Diffusion WebUI..."
        echo "🔗 WebUI буде доступний:"
        echo "   Local: http://localhost:3000"
        if [ ! -z "$RUNPOD_POD_ID" ]; then
            echo "   RunPod: https://$RUNPOD_POD_ID-3000.proxy.runpod.net"
        fi
        echo ""
        echo "💡 Для ADetailer корекції лиця використовуйте img2img з ADetailer увімкненим"
        python utils/runpod_launcher.py --launch
        ;;
    5)
        echo "📋 Приклади використання Enhanced Face Correction:"
        echo ""
        echo "🎭 Face Correction Pipeline:"
        echo "  # Демонстрація всіх можливостей"
        echo "  python scripts/demo_face_correction.py"
        echo ""
        echo "  # Одне зображення з високою якістю"
        echo "  python scripts/demo_pipeline.py --input photo.jpg --output ./results/ \\"
        echo "    --character female_portrait --model copax_realistic_xl --enhancement strong"
        echo ""
        echo "  # Пакетна обробка професійна"
        echo "  python scripts/demo_pipeline.py --batch ./photos/ --output ./results/ \\"
        echo "    --model proteus_xl --detail-model newreality_xl --quality aggressive"
        echo ""
        echo "🔧 Enhanced ADetailer:"
        echo "  # Тільки корекція лиця"
        echo "  python scripts/enhanced_adetailer.py --input photo.jpg --output enhanced.jpg \\"
        echo "    --model copax_realistic_xl --face-model face_yolov8m.pt --quality balanced"
        echo ""
        echo "🎯 Доступні SDXL моделі:"
        echo "  • epicrealism_xl - Epic реалізм (рекомендовано)"
        echo "  • copax_realistic_xl - Портретна фотографія"
        echo "  • proteus_xl - Передова деталізація"
        echo "  • newreality_xl - Експерт з деталей обличчя"
        echo "  • realvisxl_v5_lightning - Швидкий lightning"
        echo ""
        echo "👁️ Face Detection моделі:"
        echo "  • face_yolov8s.pt - Збалансований (рекомендовано)"
        echo "  • face_yolov8m.pt - Висока точність"
        echo "  • face_yolov8l.pt - Максимальна якість"
        echo ""
        echo "⚡ Рівні покращення:"
        echo "  • light - М'яке покращення"
        echo "  • medium - Збалансоване (рекомендовано)"
        echo "  • strong - Сильне покращення"
        echo "  • extreme - Максимальне покращення"
        echo ""
        echo "🎨 Типи персонажів:"
        echo "  • female_portrait - Жіночий портрет"
        echo "  • male_portrait - Чоловічий портрет"
        echo "  • child_portrait - Дитячий портрет"
        ;;
    *)
        echo "❌ Неправильний вибір"
        exit 1
        ;;
esac