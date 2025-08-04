#!/usr/bin/env python3
"""
Веб API для автоматичної обробки фотографій
Дозволяє завантажувати фото через веб-інтерфейс та скачувати результати
"""

import os
import sys
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from flask import Flask, request, jsonify, send_file, render_template_string
from werkzeug.utils import secure_filename
import threading
import queue

# Додамо шлях до scripts
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from auto_batch_processor import AutoBatchProcessor

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Глобальні змінні для управління завданнями
processor = AutoBatchProcessor()
job_queue = queue.Queue()
active_jobs = {}
completed_jobs = {}

def background_worker():
    """Фоновий процес для обробки завдань"""
    while True:
        try:
            job = job_queue.get(timeout=1)
            if job is None:  # Сигнал для зупинки
                break
                
            job_id = job['job_id']
            active_jobs[job_id]['status'] = 'processing'
            
            # Запустити обробку
            results = processor.batch_process(
                job['input_path'],
                preset=job.get('preset'),
                model=job.get('model'), 
                enhancement=job.get('enhancement'),
                max_workers=job.get('workers', 1)
            )
            
            # Створити пакет для скачування
            package_path = processor.create_download_package(results, f"job_{job_id}")
            
            # Оновити статус
            active_jobs[job_id]['status'] = 'completed'
            active_jobs[job_id]['results'] = results
            active_jobs[job_id]['package_path'] = package_path
            active_jobs[job_id]['completed_at'] = datetime.now().isoformat()
            
            # Перемістити в completed
            completed_jobs[job_id] = active_jobs.pop(job_id)
            
        except queue.Empty:
            continue
        except Exception as e:
            if job_id in active_jobs:
                active_jobs[job_id]['status'] = 'failed'
                active_jobs[job_id]['error'] = str(e)
                completed_jobs[job_id] = active_jobs.pop(job_id)

# Запустити фоновий процес
worker_thread = threading.Thread(target=background_worker, daemon=True)
worker_thread.start()

@app.route('/')
def index():
    """Головна сторінка з інтерфейсом"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>I, Model - Автоматична обробка фотографій</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .upload-area { border: 2px dashed #ccc; padding: 20px; text-align: center; margin: 20px 0; }
            .upload-area:hover { border-color: #999; }
            .form-group { margin: 10px 0; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            select, input, button { padding: 8px; margin: 5px; }
            button { background: #007cba; color: white; border: none; cursor: pointer; }
            button:hover { background: #005a87; }
            .job-status { margin: 20px 0; padding: 10px; border-radius: 5px; }
            .processing { background: #fff3cd; }
            .completed { background: #d4edda; }
            .failed { background: #f8d7da; }
        </style>
    </head>
    <body>
        <h1>🎨 I, Model - Автоматична обробка фотографій</h1>
        
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <p>📁 Клікніть або перетягніть фото для завантаження</p>
                <input type="file" id="fileInput" name="files" multiple accept="image/*" style="display: none;">
            </div>
            
            <div class="form-group">
                <label>Стиль обробки:</label>
                <select name="preset">
                    <option value="professional_headshot">Професійний портрет</option>
                    <option value="artistic_portrait">Артистичний портрет</option>
                    <option value="natural_candid">Природній стиль</option>
                    <option value="glamour_portrait">Гламурний портрет</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>Модель SDXL:</label>
                <select name="model">
                    <option value="epicrealism_xl">EpiCRealism XL</option>
                    <option value="realvisxl_v5_lightning">RealVisXL Lightning</option>
                    <option value="juggernaut_xl_v9">Juggernaut XL v9</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>Рівень покращення:</label>
                <select name="enhancement">
                    <option value="light">Легкий</option>
                    <option value="medium" selected>Середній</option>
                    <option value="strong">Сильний</option>
                    <option value="extreme">Максимальний</option>
                </select>
            </div>
            
            <button type="submit">🚀 Почати обробку</button>
        </form>
        
        <div id="jobStatus"></div>
        
        <script>
            document.getElementById('uploadForm').onsubmit = async function(e) {
                e.preventDefault();
                const formData = new FormData(this);
                
                try {
                    const response = await fetch('/upload', { method: 'POST', body: formData });
                    const result = await response.json();
                    
                    if (result.success) {
                        pollJobStatus(result.job_id);
                    } else {
                        alert('Помилка: ' + result.error);
                    }
                } catch (error) {
                    alert('Помилка завантаження: ' + error);
                }
            };
            
            async function pollJobStatus(jobId) {
                const statusDiv = document.getElementById('jobStatus');
                
                while (true) {
                    try {
                        const response = await fetch(`/job/${jobId}`);
                        const job = await response.json();
                        
                        statusDiv.innerHTML = `
                            <div class="job-status ${job.status}">
                                <h3>Завдання ${jobId}</h3>
                                <p>Статус: ${getStatusText(job.status)}</p>
                                ${job.progress ? `<p>Прогрес: ${job.progress}</p>` : ''}
                                ${job.status === 'completed' ? 
                                    `<p><a href="/download/${jobId}">📦 Скачати результати</a></p>` : ''}
                                ${job.error ? `<p>Помилка: ${job.error}</p>` : ''}
                            </div>
                        `;
                        
                        if (job.status === 'completed' || job.status === 'failed') {
                            break;
                        }
                        
                        await new Promise(resolve => setTimeout(resolve, 2000));
                    } catch (error) {
                        console.error('Помилка отримання статусу:', error);
                        break;
                    }
                }
            }
            
            function getStatusText(status) {
                const texts = {
                    'queued': 'В черзі',
                    'processing': 'Обробляється...',
                    'completed': 'Завершено ✅',
                    'failed': 'Помилка ❌'
                };
                return texts[status] || status;
            }
        </script>
    </body>
    </html>
    """
    return html

@app.route('/upload', methods=['POST'])
def upload_files():
    """Завантажити файли та почати обробку"""
    try:
        if 'files' not in request.files:
            return jsonify({'success': False, 'error': 'Файли не вибрано'})
        
        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            return jsonify({'success': False, 'error': 'Файли не вибрано'})
        
        # Створити унікальний job ID
        job_id = str(uuid.uuid4())[:8]
        job_dir = os.path.join(processor.config['input_dir'], f'job_{job_id}')
        os.makedirs(job_dir, exist_ok=True)
        
        # Зберегти файли
        saved_files = []
        for file in files:
            if file.filename:
                filename = secure_filename(file.filename)
                filepath = os.path.join(job_dir, filename)
                file.save(filepath)
                saved_files.append(filepath)
        
        if not saved_files:
            return jsonify({'success': False, 'error': 'Не вдалося зберегти файли'})
        
        # Створити завдання
        job = {
            'job_id': job_id,
            'input_path': job_dir,
            'preset': request.form.get('preset', 'professional_headshot'),
            'model': request.form.get('model', 'epicrealism_xl'),
            'enhancement': request.form.get('enhancement', 'medium'),
            'workers': 1,  # Для веб API використовуємо 1 worker
            'created_at': datetime.now().isoformat(),
            'files_count': len(saved_files)
        }
        
        # Додати в чергу
        active_jobs[job_id] = {
            'status': 'queued',
            'created_at': job['created_at'],
            'files_count': job['files_count'],
            'params': {k: v for k, v in job.items() if k not in ['job_id', 'input_path']}
        }
        
        job_queue.put(job)
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'files_count': len(saved_files),
            'message': f'Завдання {job_id} створено. Обробляється {len(saved_files)} файлів.'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/job/<job_id>')
def get_job_status(job_id):
    """Отримати статус завдання"""
    if job_id in active_jobs:
        return jsonify(active_jobs[job_id])
    elif job_id in completed_jobs:
        job = completed_jobs[job_id].copy()
        # Не повертати великі дані
        if 'results' in job:
            job['results_count'] = len(job['results'])
            job['successful_count'] = sum(1 for r in job['results'] if r['success'])
            del job['results']
        return jsonify(job)
    else:
        return jsonify({'error': 'Завдання не знайдено'}), 404

@app.route('/download/<job_id>')
def download_results(job_id):
    """Скачати результати завдання"""
    if job_id not in completed_jobs:
        return jsonify({'error': 'Завдання не знайдено або не завершено'}), 404
    
    job = completed_jobs[job_id]
    if 'package_path' not in job or not os.path.exists(job['package_path']):
        return jsonify({'error': 'Пакет результатів не знайдено'}), 404
    
    return send_file(
        job['package_path'],
        as_attachment=True,
        download_name=f'enhanced_photos_{job_id}.zip'
    )

@app.route('/jobs')
def list_jobs():
    """Список всіх завдань"""
    all_jobs = {}
    all_jobs.update(active_jobs)
    all_jobs.update(completed_jobs)
    
    return jsonify(all_jobs)

if __name__ == '__main__':
    print("🌐 Запуск Web API для автоматичної обробки фотографій")
    print("📱 Відкрийте в браузері: http://localhost:5000")
    print("🎯 API endpoints:")
    print("  - POST /upload - завантажити фото")
    print("  - GET /job/<id> - статус завдання") 
    print("  - GET /download/<id> - скачати результати")
    
    app.run(host='0.0.0.0', port=5000, debug=True)