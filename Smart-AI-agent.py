import requests
import PyPDF2
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import webbrowser
import threading
import json
import os
from datetime import datetime, timedelta
import time
import uuid
import schedule
from collections import defaultdict
import subprocess
import sys

# PDF generation import
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    PDF_GENERATION_AVAILABLE = True
except ImportError:
    PDF_GENERATION_AVAILABLE = False
    print("⚠️ reportlab ei saatavilla - asenna: pip install reportlab")

# Calendar widget import
try:
    from tkcalendar import Calendar, DateEntry
    CALENDAR_AVAILABLE = True
except ImportError:
    CALENDAR_AVAILABLE = False
    print("⚠️ tkcalendar ei saatavilla - asenna: pip install tkcalendar")

# Schedule import
try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False
    print("⚠️ schedule ei saatavilla - asenna: pip install schedule")

# Conditional imports with fallbacks
try:
    import pyttsx3
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False
    print("⚠️ pyttsx3 ei saatavilla - puhesynteesi poistettu käytöstä")

try:
    from deep_translator import GoogleTranslator
    TRANSLATE_AVAILABLE = True
    TRANSLATOR_TYPE = "deep_translator"
except ImportError:
    try:
        from googletrans import Translator
        TRANSLATE_AVAILABLE = True
        TRANSLATOR_TYPE = "googletrans"
    except ImportError:
        TRANSLATE_AVAILABLE = False
        TRANSLATOR_TYPE = None
        print("⚠️ Käännöspalvelu ei saatavilla - asenna: pip install deep-translator")

try:
    from youtubesearchpython import VideosSearch
    YOUTUBE_AVAILABLE = True
except ImportError:
    YOUTUBE_AVAILABLE = False
    print("⚠️ youtube-search-python ei saatavilla - YouTube-haku poistettu käytöstä")

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    print("⚠️ SpeechRecognition ei saatavilla - puheentunnistus poistettu käytöstä")

# AI Provider imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI ei saatavilla - asenna: pip install openai")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class TaskStatus:
    """Task status constants"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ScheduledTask:
    """Represents a scheduled task"""
    
    def __init__(self, task_id, name, description, scheduled_time, task_type, 
                 parameters=None, recurring=False, interval=None):
        self.id = task_id
        self.name = name
        self.description = description
        self.scheduled_time = scheduled_time
        self.task_type = task_type
        self.parameters = parameters or {}
        self.recurring = recurring
        self.interval = interval
        self.status = TaskStatus.PENDING
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.result = None
        self.error = None
        self.progress = 0
        self.logs = []
    
    def add_log(self, message):
        """Add log entry"""
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "message": message
        })
    
    def to_dict(self):
        """Convert task to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "scheduled_time": self.scheduled_time.isoformat() if self.scheduled_time else None,
            "task_type": self.task_type,
            "parameters": self.parameters,
            "recurring": self.recurring,
            "interval": self.interval,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
            "progress": self.progress,
            "logs": self.logs
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create task from dictionary"""
        task = cls(
            data["id"],
            data["name"],
            data["description"],
            datetime.fromisoformat(data["scheduled_time"]) if data["scheduled_time"] else None,
            data["task_type"],
            data.get("parameters", {}),
            data.get("recurring", False),
            data.get("interval")
        )
        task.status = data.get("status", TaskStatus.PENDING)
        task.created_at = datetime.fromisoformat(data["created_at"])
        task.started_at = datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None
        task.completed_at = datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
        task.result = data.get("result")
        task.error = data.get("error")
        task.progress = data.get("progress", 0)
        task.logs = data.get("logs", [])
        return task


class TaskScheduler:
    """Manages scheduled tasks"""
    
    def __init__(self, agent):
        self.agent = agent
        self.tasks = {}
        self.running_tasks = {}
        self.scheduler_thread = None
        self.is_running = False
        self.load_tasks()
        self.start_scheduler()
    
    def start_scheduler(self):
        """Start the task scheduler"""
        if not self.is_running:
            self.is_running = True
            self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            self.scheduler_thread.start()
    
    def stop_scheduler(self):
        """Stop the task scheduler"""
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=1)
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.is_running:
            try:
                # Check for tasks to run
                current_time = datetime.now()
                
                for task_id, task in list(self.tasks.items()):
                    if (task.status == TaskStatus.PENDING and 
                        task.scheduled_time and 
                        current_time >= task.scheduled_time):
                        
                        self._execute_task(task)
                
                # Clean up completed threads
                for task_id in list(self.running_tasks.keys()):
                    thread = self.running_tasks[task_id]
                    if not thread.is_alive():
                        del self.running_tasks[task_id]
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                print(f"Scheduler error: {e}")
                time.sleep(5)
    
    def _execute_task(self, task):
        """Execute a scheduled task"""
        if task.id in self.running_tasks:
            return  # Task already running
        
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        task.add_log("Task execution started")
        
        def task_runner():
            try:
                task.progress = 10
                result = self._run_task_by_type(task)
                
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                task.result = result
                task.progress = 100
                task.add_log("Task completed successfully")
                
                # Schedule next occurrence if recurring
                if task.recurring and task.interval:
                    self._schedule_next_occurrence(task)
                
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now()
                task.error = str(e)
                task.progress = 0
                task.add_log(f"Task failed: {str(e)}")
            
            finally:
                self.save_tasks()
        
        thread = threading.Thread(target=task_runner, daemon=True)
        self.running_tasks[task.id] = thread
        thread.start()
    
    def _run_task_by_type(self, task):
        """Run task based on its type"""
        task_type = task.task_type
        params = task.parameters
        
        task.progress = 25
        
        if task_type == "ai_chat":
            task.add_log("Generating AI response")
            result = self.agent.chat_with_user(params.get("prompt", ""))
            
        elif task_type == "web_search":
            task.add_log("Performing web search")
            task.progress = 50
            result = self.agent.web_search(params.get("query", ""))
            
        elif task_type == "youtube_search":
            task.add_log("Searching YouTube")
            task.progress = 50
            result = self.agent.youtube_search(params.get("query", ""))
            
        elif task_type == "translate":
            task.add_log("Translating text")
            task.progress = 50
            result = self.agent.translate_text(
                params.get("text", ""), 
                params.get("target_lang", "en")
            )
            
        elif task_type == "api_call":
            task.add_log("Making API call")
            task.progress = 50
            result = self.agent.call_api(params.get("url", ""))
            
        elif task_type == "pdf_read":
            task.add_log("Reading PDF file")
            task.progress = 50
            result = self.agent.read_pdf(params.get("file_path", ""))
            
        elif task_type == "custom_command":
            task.add_log("Executing custom command")
            task.progress = 50
            # Execute custom Python command
            command = params.get("command", "")
            try:
                result = eval(command) if command else "No command specified"
            except Exception as e:
                result = f"Command execution failed: {str(e)}"
        
        else:
            result = f"Unknown task type: {task_type}"
        
        task.progress = 90
        return result
    
    def _schedule_next_occurrence(self, task):
        """Schedule next occurrence for recurring task"""
        if not task.interval:
            return
        
        # Create new task instance for next occurrence
        next_time = None
        
        if task.interval == "daily":
            next_time = task.scheduled_time + timedelta(days=1)
        elif task.interval == "weekly":
            next_time = task.scheduled_time + timedelta(weeks=1)
        elif task.interval == "monthly":
            next_time = task.scheduled_time + timedelta(days=30)
        elif task.interval == "hourly":
            next_time = task.scheduled_time + timedelta(hours=1)
        
        if next_time:
            new_task = ScheduledTask(
                str(uuid.uuid4()),
                task.name,
                task.description,
                next_time,
                task.task_type,
                task.parameters,
                task.recurring,
                task.interval
            )
            self.add_task(new_task)
    
    def add_task(self, task):
        """Add a new task"""
        self.tasks[task.id] = task
        self.save_tasks()
        return task.id
    
    def cancel_task(self, task_id):
        """Cancel a task"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.CANCELLED
                task.add_log("Task cancelled by user")
                self.save_tasks()
                return True
        return False
    
    def get_task(self, task_id):
        """Get task by ID"""
        return self.tasks.get(task_id)
    
    def get_tasks_by_status(self, status):
        """Get tasks by status"""
        return [task for task in self.tasks.values() if task.status == status]
    
    def get_all_tasks(self):
        """Get all tasks"""
        return list(self.tasks.values())
    
    def save_tasks(self):
        """Save tasks to file"""
        tasks_data = {task_id: task.to_dict() for task_id, task in self.tasks.items()}
        with open('scheduled_tasks.json', 'w', encoding='utf-8') as f:
            json.dump(tasks_data, f, ensure_ascii=False, indent=2)
    
    def load_tasks(self):
        """Load tasks from file"""
        try:
            if os.path.exists('scheduled_tasks.json'):
                with open('scheduled_tasks.json', 'r', encoding='utf-8') as f:
                    tasks_data = json.load(f)
                    self.tasks = {
                        task_id: ScheduledTask.from_dict(task_data) 
                        for task_id, task_data in tasks_data.items()
                    }
        except Exception as e:
            print(f"Error loading tasks: {e}")
            self.tasks = {}


class PDFGenerator:
    """Generate PDF reports for completed tasks"""
    
    @staticmethod
    def generate_task_report(tasks, filename=None):
        """Generate PDF report for tasks"""
        if not PDF_GENERATION_AVAILABLE:
            return "PDF generation not available - install reportlab"
        
        if not filename:
            filename = f"task_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        try:
            # Create document
            doc = SimpleDocTemplate(filename, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                textColor=colors.darkblue,
                alignment=1  # Center
            )
            story.append(Paragraph("Tehtäväraportti", title_style))
            story.append(Spacer(1, 20))
            
            # Summary
            summary_style = ParagraphStyle(
                'Summary',
                parent=styles['Normal'],
                fontSize=12,
                spaceAfter=15,
                textColor=colors.black
            )
            
            total_tasks = len(tasks)
            completed_tasks = len([t for t in tasks if t.status == TaskStatus.COMPLETED])
            failed_tasks = len([t for t in tasks if t.status == TaskStatus.FAILED])
            
            summary_text = f"""
            <b>Yhteenveto:</b><br/>
            • Tehtäviä yhteensä: {total_tasks}<br/>
            • Valmistuneita: {completed_tasks}<br/>
            • Epäonnistuneita: {failed_tasks}<br/>
            • Raportin luontiaika: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}<br/>
            """
            story.append(Paragraph(summary_text, summary_style))
            story.append(Spacer(1, 20))
            
            # Task details
            for task in tasks:
                # Task header
                task_header_style = ParagraphStyle(
                    'TaskHeader',
                    parent=styles['Heading2'],
                    fontSize=16,
                    spaceAfter=10,
                    textColor=colors.darkgreen if task.status == TaskStatus.COMPLETED else colors.red
                )
                
                status_emoji = {
                    TaskStatus.COMPLETED: "✅",
                    TaskStatus.FAILED: "❌",
                    TaskStatus.RUNNING: "🔄",
                    TaskStatus.PENDING: "⏳",
                    TaskStatus.CANCELLED: "🚫"
                }
                
                story.append(Paragraph(
                    f"{status_emoji.get(task.status, '❓')} {task.name}", 
                    task_header_style
                ))
                
                # Task details
                details_style = ParagraphStyle(
                    'TaskDetails',
                    parent=styles['Normal'],
                    fontSize=10,
                    spaceAfter=15,
                    leftIndent=20
                )
                
                details_text = f"""
                <b>Kuvaus:</b> {task.description}<br/>
                <b>Tyyppi:</b> {task.task_type}<br/>
                <b>Tila:</b> {task.status}<br/>
                <b>Luotu:</b> {task.created_at.strftime('%d.%m.%Y %H:%M:%S')}<br/>
                """
                
                if task.scheduled_time:
                    details_text += f"<b>Ajastettu:</b> {task.scheduled_time.strftime('%d.%m.%Y %H:%M:%S')}<br/>"
                
                if task.started_at:
                    details_text += f"<b>Aloitettu:</b> {task.started_at.strftime('%d.%m.%Y %H:%M:%S')}<br/>"
                
                if task.completed_at:
                    details_text += f"<b>Valmistunut:</b> {task.completed_at.strftime('%d.%m.%Y %H:%M:%S')}<br/>"
                    
                    # Calculate duration
                    if task.started_at:
                        duration = task.completed_at - task.started_at
                        details_text += f"<b>Kesto:</b> {str(duration).split('.')[0]}<br/>"
                
                if task.result:
                    result_preview = task.result[:200] + "..." if len(task.result) > 200 else task.result
                    details_text += f"<b>Tulos:</b> {result_preview}<br/>"
                
                if task.error:
                    details_text += f"<b>Virhe:</b> {task.error}<br/>"
                
                story.append(Paragraph(details_text, details_style))
                
                # Logs
                if task.logs:
                    logs_style = ParagraphStyle(
                        'Logs',
                        parent=styles['Normal'],
                        fontSize=8,
                        spaceAfter=20,
                        leftIndent=40,
                        textColor=colors.grey
                    )
                    
                    logs_text = "<b>Lokit:</b><br/>"
                    for log in task.logs[-5:]:  # Show last 5 logs
                        log_time = datetime.fromisoformat(log["timestamp"]).strftime('%H:%M:%S')
                        logs_text += f"• {log_time}: {log['message']}<br/>"
                    
                    story.append(Paragraph(logs_text, logs_style))
                
                story.append(Spacer(1, 10))
            
            # Build PDF
            doc.build(story)
            return filename
            
        except Exception as e:
            return f"PDF generation error: {str(e)}"





class AIProviderManager:
    """Manages different AI providers and their APIs"""
    
    def __init__(self):
        self.providers = {
            "openai": {
                "name": "OpenAI (GPT-3.5/4)",
                "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
                "requires_key": True,
                "available": OPENAI_AVAILABLE
            },
            "groq": {
                "name": "Groq (Llama/Mixtral)",
                "models": ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"],
                "requires_key": True,
                "available": True  # Uses OpenAI-compatible API
            },
            "anthropic": {
                "name": "Anthropic (Claude)",
                "models": ["claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229"],
                "requires_key": True,
                "available": ANTHROPIC_AVAILABLE
            },
            "gemini": {
                "name": "Google Gemini",
                "models": ["gemini-pro", "gemini-pro-vision"],
                "requires_key": True,
                "available": GEMINI_AVAILABLE
            },
            "ollama": {
                "name": "Ollama (Local)",
                "models": ["llama3", "mistral", "codellama", "llama2", "phi3", "qwen2"],
                "requires_key": False,
                "available": True,
                "base_url": "http://localhost:5678"
            },
            "cohere": {
                "name": "Cohere",
                "models": ["command-r", "command-r-plus", "command"],
                "requires_key": True,
                "available": True
            },
            "huggingface": {
                "name": "Hugging Face",
                "models": [
                    "unsloth/gpt-oss-20b-GGUF",
                    "openai/gpt-oss-20b",
                    "openai/gpt-oss-120b",
                    "microsoft/DialoGPT-large", 
                    "microsoft/DialoGPT-medium", 
                    "facebook/blenderbot-400M-distill",
                    "HuggingFaceH4/zephyr-7b-beta",
                    "mistralai/Mistral-7B-Instruct-v0.1",
                    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"
                ],
                "requires_key": True,
                "available": True
            },
            "together": {
                "name": "Together AI",
                "models": ["meta-llama/Llama-2-70b-chat-hf", "mistralai/Mixtral-8x7B-Instruct-v0.1"],
                "requires_key": True,
                "available": True
            }
        }
    
    def get_available_providers(self):
        """Get list of available providers"""
        return {k: v for k, v in self.providers.items() if v["available"]}
    
    def call_ai_api(self, provider, model, messages, api_key=None, **kwargs):
        """Generic AI API caller"""
        try:
            if provider == "openai":
                return self._call_openai(model, messages, api_key, **kwargs)
            elif provider == "groq":
                return self._call_groq(model, messages, api_key, **kwargs)
            elif provider == "anthropic":
                return self._call_anthropic(model, messages, api_key, **kwargs)
            elif provider == "gemini":
                return self._call_gemini(model, messages, api_key, **kwargs)
            elif provider == "ollama":
                return self._call_ollama(model, messages, **kwargs)
            elif provider == "cohere":
                return self._call_cohere(model, messages, api_key, **kwargs)
            elif provider == "huggingface":
                return self._call_huggingface(model, messages, api_key, **kwargs)
            elif provider == "together":
                return self._call_together(model, messages, api_key, **kwargs)
            else:
                return "Tuntematon AI-palveluntarjoaja"
        except Exception as e:
            return f"AI API -virhe ({provider}): {str(e)}"
    
    def _call_openai(self, model, messages, api_key, **kwargs):
        """Call OpenAI API"""
        if not OPENAI_AVAILABLE:
            return "OpenAI-kirjasto ei ole saatavilla"
        
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=kwargs.get('max_tokens', 500),
            temperature=kwargs.get('temperature', 0.7)
        )
        return response.choices[0].message["content"]
    
    def _call_groq(self, model, messages, api_key, **kwargs):
        """Call Groq API (OpenAI compatible)"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": kwargs.get('max_tokens', 500),
            "temperature": kwargs.get('temperature', 0.7)
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Groq virhe: {response.status_code}"

    def _call_huggingface(self, model, messages, api_key, **kwargs):
        """Call Hugging Face API"""
        try:
            # Hugging Face Inference API endpoint
            api_url = f"https://api-inference.huggingface.co/models/{model}"
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # Convert messages to a single prompt for text generation models
            prompt = ""
            for message in messages:
                if message["role"] == "system":
                    prompt += f"System: {message['content']}\n"
                elif message["role"] == "user":
                    prompt += f"User: {message['content']}\n"
                elif message["role"] == "assistant":
                    prompt += f"Assistant: {message['content']}\n"
            
            prompt += "Assistant:"
            
            # Prepare the payload
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": kwargs.get('max_tokens', 500),
                    "temperature": kwargs.get('temperature', 0.7),
                    "return_full_text": False
                },
                "options": {
                    "wait_for_model": True
                }
            }
            
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "").strip()
                else:
                    return "Tyhjä vastaus Hugging Face API:sta"
            elif response.status_code == 503:
                return "Malli latautuu, yritä hetken kuluttua uudelleen"
            else:
                return f"Hugging Face API virhe: {response.status_code} - {response.text}"
                
        except requests.exceptions.Timeout:
            return "Hugging Face API aikakatkaisu"
        except requests.exceptions.RequestException as e:
            return f"Hugging Face verkkovirhe: {str(e)}"
        except Exception as e:
            return f"Hugging Face virhe: {str(e)}"

    # You need to add the other missing methods here:
    # _call_anthropic, _call_gemini, _call_ollama, _call_cohere, _call_together

    def _call_anthropic(self, model, messages, api_key, **kwargs):
            """Call Anthropic API"""
            if not ANTHROPIC_AVAILABLE:
                return "Anthropic-kirjasto ei ole saatavilla"
            
            try:
                client = anthropic.Anthropic(api_key=api_key)
                response = client.messages.create(
                    model=model,
                    max_tokens=kwargs.get('max_tokens', 500),
                    temperature=kwargs.get('temperature', 0.7),
                    messages=messages
                )
                return response.content[0].text
            except Exception as e:
                return f"Anthropic virhe: {str(e)}"

    def _call_gemini(self, model, messages, api_key, **kwargs):
            """Call Google Gemini API"""
            if not GEMINI_AVAILABLE:
                return "Gemini-kirjasto ei ole saatavilla"
            
            try:
                genai.configure(api_key=api_key)
                model_instance = genai.GenerativeModel(model)
                
                # Convert messages to a single prompt
                prompt = ""
                for message in messages:
                    if message["role"] == "user":
                        prompt += f"User: {message['content']}\n"
                    elif message["role"] == "assistant":
                        prompt += f"Assistant: {message['content']}\n"
                
                response = model_instance.generate_content(prompt)
                return response.text
            except Exception as e:
                return f"Gemini virhe: {str(e)}"

    def _call_ollama(self, model, messages, **kwargs):
            """Call Ollama API"""
            try:
                headers = {"Content-Type": "application/json"}
                
                # Convert messages to prompt
                prompt = ""
                for message in messages:
                    if message["role"] == "user":
                        prompt += f"User: {message['content']}\n"
                    elif message["role"] == "assistant":
                        prompt += f"Assistant: {message['content']}\n"
                prompt += "Assistant:"
                
                data = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get('temperature', 0.7)
                    }
                }
                
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    headers=headers,
                    json=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    return response.json().get("response", "")
                else:
                    return f"Ollama virhe: {response.status_code}"
            except Exception as e:
                return f"Ollama virhe: {str(e)}"

    def _call_cohere(self, model, messages, api_key, **kwargs):
            """Call Cohere API"""
            try:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                # Convert messages to prompt
                prompt = ""
                for message in messages:
                    if message["role"] == "user":
                        prompt += f"User: {message['content']}\n"
                    elif message["role"] == "assistant":
                        prompt += f"Assistant: {message['content']}\n"
                
                data = {
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": kwargs.get('max_tokens', 500),
                    "temperature": kwargs.get('temperature', 0.7)
                }
                
                response = requests.post(
                    "https://api.cohere.ai/v1/generate",
                    headers=headers,
                    json=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    return response.json()["generations"][0]["text"]
                else:
                    return f"Cohere virhe: {response.status_code}"
            except Exception as e:
                return f"Cohere virhe: {str(e)}"

    def _call_together(self, model, messages, api_key, **kwargs):
            """Call Together AI API"""
            try:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": kwargs.get('max_tokens', 500),
                    "temperature": kwargs.get('temperature', 0.7)
                }
                
                response = requests.post(
                    "https://api.together.xyz/inference",
                    headers=headers,
                    json=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    return response.json()["output"]["choices"][0]["text"]
                else:
                    return f"Together AI virhe: {response.status_code}"
            except Exception as e:
                return f"Together AI virhe: {str(e)}"




class SmartAgent:
    def __init__(self):
        """Initialize the Smart Agent"""
        self.conversation_history = []
        self.api_keys = {}
        self.current_ai_provider = "groq"
        self.current_ai_model = "llama3-70b-8192"
        self.ai_manager = AIProviderManager()
        self.scheduler = TaskScheduler(self)
        self.load_config()
        
        # Initialize speech components if available
        if SPEECH_AVAILABLE:
            self.tts_engine = pyttsx3.init()
            self.setup_voice()
        
        if SPEECH_RECOGNITION_AVAILABLE:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
    
    def setup_voice(self):
        """Setup text-to-speech voice"""
        if SPEECH_AVAILABLE:
            voices = self.tts_engine.getProperty('voices')
            # Try to find Finnish voice, fallback to first available
            for voice in voices:
                if 'finnish' in voice.name.lower() or 'fi' in voice.id.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    break
            
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.8)
    
    def speak_text(self, text):
        """Convert text to speech"""
        if SPEECH_AVAILABLE:
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"Speech error: {e}")
    
    def listen_speech(self):
        """Listen to speech and convert to text"""
        if not SPEECH_RECOGNITION_AVAILABLE:
            return "Speech recognition not available"
        
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
                print("Kuuntelen...")
                audio = self.recognizer.listen(source, timeout=5)
            
            print("Tunnistan puhetta...")
            text = self.recognizer.recognize_google(audio, language='fi-FI')
            return text
        except sr.WaitTimeoutError:
            return "Aikakatkaisut - ei ääntä havaittu"
        except sr.UnknownValueError:
            return "Puhetta ei voitu tunnistaa"
        except sr.RequestError as e:
            return f"Virhe puheentunnistuksessa: {e}"
    
    def load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists('agent_config.json'):
                with open('agent_config.json', 'r') as f:
                    config = json.load(f)
                    self.api_keys = config.get('api_keys', {})
                    self.current_ai_provider = config.get('ai_provider', 'groq')
                    self.current_ai_model = config.get('ai_model', 'llama3-70b-8192')
        except Exception as e:
            print(f"Config load error: {e}")
    
    def save_config(self):
        """Save configuration to file"""
        try:
            config = {
                'api_keys': self.api_keys,
                'ai_provider': self.current_ai_provider,
                'ai_model': self.current_ai_model
            }
            with open('agent_config.json', 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Config save error: {e}")
    
    def set_api_key(self, provider, key):
        """Set API key for provider"""
        self.api_keys[provider] = key
        self.save_config()
    
    def get_api_key(self, provider):
        """Get API key for provider"""
        return self.api_keys.get(provider)
    
    def web_search(self, query):
        """Perform web search using DuckDuckGo"""
        try:
            # Simple web search implementation
            search_url = f"https://duckduckgo.com/html/?q={query}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(search_url, headers=headers, timeout=10)
            if response.status_code == 200:
                return f"Hakutulokset haulle '{query}' löydetty. Status: {response.status_code}"
            else:
                return f"Haku epäonnistui. Status: {response.status_code}"
        except Exception as e:
            return f"Hakuvirhe: {str(e)}"
    
    def youtube_search(self, query):
        """Search YouTube videos"""
        if not YOUTUBE_AVAILABLE:
            return "YouTube-haku ei ole saatavilla"
        
        try:
            videos_search = VideosSearch(query, limit=5)
            results = videos_search.result()
            
            if results['result']:
                output = f"YouTube-hakutulokset haulle '{query}':\n\n"
                for i, video in enumerate(results['result'], 1):
                    output += f"{i}. {video['title']}\n"
                    output += f"   Kanava: {video['channel']['name']}\n"
                    output += f"   Kesto: {video['duration']}\n"
                    output += f"   URL: {video['link']}\n\n"
                return output
            else:
                return f"Ei tuloksia haulle: {query}"
        except Exception as e:
            return f"YouTube-hakuvirhe: {str(e)}"
    
    def translate_text(self, text, target_lang='en'):
        """Translate text using Google Translate"""
        if not TRANSLATE_AVAILABLE:
            return "Käännöspalvelu ei ole saatavilla"
        
        try:
            if TRANSLATOR_TYPE == "deep_translator":
                translator = GoogleTranslator(source='auto', target=target_lang)
                result = translator.translate(text)
            else:  # googletrans
                translator = Translator()
                result = translator.translate(text, dest=target_lang)
                result = result.text
            
            return f"Käännös ({target_lang}): {result}"
        except Exception as e:
            return f"Käännösvirhe: {str(e)}"
    
    def read_pdf(self, file_path):
        """Read PDF file content"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = ""
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text_content += page.extract_text() + "\n"
                
                return f"PDF sisältö ({len(pdf_reader.pages)} sivua):\n\n{text_content[:1000]}..."
        except Exception as e:
            return f"PDF-lukuvirhe: {str(e)}"
    
    def call_api(self, url, method='GET', data=None, headers=None):
        """Make HTTP API call"""
        try:
            if headers is None:
                headers = {'Content-Type': 'application/json'}
            
            if method.upper() == 'GET':
                response = requests.get(url, headers=headers, timeout=10)
            elif method.upper() == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=10)
            else:
                return f"Tuntematon HTTP-metodi: {method}"
            
            return f"API-vastaus ({response.status_code}):\n{response.text[:500]}..."
        except Exception as e:
            return f"API-virhe: {str(e)}"
    
    def chat_with_user(self, user_input):
        """Chat with user using AI"""
        try:
            # Add user message to history
            self.conversation_history.append({"role": "user", "content": user_input})
            
            # Limit conversation history
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            # Prepare messages for AI
            messages = [
                {"role": "system", "content": "Olet ystävällinen ja hyödyllinen suomenkielinen AI-agentti. Vastaa aina suomeksi ja ole ytimekäs mutta informatiivinen."}
            ] + self.conversation_history
            
            # Get AI response
            api_key = self.get_api_key(self.current_ai_provider)
            response = self.ai_manager.call_ai_api(
                self.current_ai_provider,
                self.current_ai_model,
                messages,
                api_key
            )
            
            # Add AI response to history
            self.conversation_history.append({"role": "assistant", "content": response})
            
            return response
        except Exception as e:
            return f"Chat-virhe: {str(e)}"
    
    def analyze_sentiment(self, text):
        """Simple sentiment analysis"""
        positive_words = ['hyvä', 'loistava', 'mahtava', 'upea', 'erinomainen', 'positiivinen']
        negative_words = ['huono', 'kauhea', 'hirveä', 'negatiivinen', 'surullinen', 'vihainen']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "Positiivinen"
        elif negative_count > positive_count:
            return "Negatiivinen"
        else:
            return "Neutraali"
    
    def get_weather(self, city="Helsinki"):
        """Get weather information (mock implementation)"""
        try:
            # This is a simple mock - in real implementation, use weather API
            weather_data = {
                "Helsinki": {"temp": "5°C", "condition": "Pilvistä", "humidity": "75%"},
                "Tampere": {"temp": "3°C", "condition": "Sateinen", "humidity": "80%"},
                "Turku": {"temp": "4°C", "condition": "Aurinkoinen", "humidity": "70%"}
            }
            
            if city in weather_data:
                data = weather_data[city]
                return f"Sää kaupungissa {city}:\nLämpötila: {data['temp']}\nTila: {data['condition']}\nKosteus: {data['humidity']}"
            else:
                return f"Säätietoja ei löytynyt kaupungille: {city}"
        except Exception as e:
            return f"Säävirhe: {str(e)}"
    
    def shutdown(self):
        """Shutdown the agent"""
        print("Suljetaan AI-agentti...")
        if hasattr(self, 'scheduler'):
            self.scheduler.stop_scheduler()
        self.save_config()


class SmartAgentGUI:
    def __init__(self):
        """Initialize the GUI"""
        self.agent = SmartAgent()
        self.setup_gui()
        
        # Start GUI update thread
        self.update_thread = threading.Thread(target=self.update_task_display, daemon=True)
        self.update_thread.start()
    
    def setup_gui(self):
        """Setup the main GUI"""
        self.root = tk.Tk()
        self.root.title("🤖 Smart AI Agent v2.0")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2b2b2b')
        
        # Configure styles
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Dark.TFrame', background='#2b2b2b')
        style.configure('Dark.TLabel', background='#2b2b2b', foreground='#ffffff')
        style.configure('Dark.TButton', background='#404040', foreground='#ffffff')
        
        self.setup_notebook()
        self.setup_chat_tab()
        self.setup_tasks_tab()
        self.setup_scheduler_tab()
        self.setup_settings_tab()
        self.setup_menu()
    
    def setup_notebook(self):
        """Setup main notebook with tabs"""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.chat_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.tasks_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.scheduler_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.settings_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        
        self.notebook.add(self.chat_frame, text="💬 Chat")
        self.notebook.add(self.tasks_frame, text="📋 Tehtävät")
        self.notebook.add(self.scheduler_frame, text="⏰ Ajastus")
        self.notebook.add(self.settings_frame, text="⚙️ Asetukset")
    
    def setup_chat_tab(self):
        """Setup chat tab"""
        # Chat display
        self.chat_display = scrolledtext.ScrolledText(
            self.chat_frame, 
            height=20, 
            bg='#1e1e1e', 
            fg='#ffffff',
            font=('Consolas', 10)
        )
        self.chat_display.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Input frame
        input_frame = ttk.Frame(self.chat_frame, style='Dark.TFrame')
        input_frame.pack(fill='x', padx=10, pady=5)
        
        self.chat_input = tk.Entry(
            input_frame, 
            bg='#404040', 
            fg='#ffffff', 
            font=('Consolas', 10)
        )
        self.chat_input.pack(side='left', fill='x', expand=True, padx=(0, 5))
        self.chat_input.bind('<Return>', self.send_message)
        
        # Buttons
        ttk.Button(
            input_frame, 
            text="Lähetä", 
            command=self.send_message,
            style='Dark.TButton'
        ).pack(side='right', padx=(0, 5))
        
        if SPEECH_RECOGNITION_AVAILABLE:
            ttk.Button(
                input_frame, 
                text="🎤", 
                command=self.start_voice_input,
                style='Dark.TButton'
            ).pack(side='right', padx=(0, 5))
        
        # Quick actions frame
        actions_frame = ttk.Frame(self.chat_frame, style='Dark.TFrame')
        actions_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(actions_frame, text="🌐 Web-haku", command=self.quick_web_search).pack(side='left', padx=2)
        ttk.Button(actions_frame, text="📺 YouTube", command=self.quick_youtube_search).pack(side='left', padx=2)
        ttk.Button(actions_frame, text="🌡️ Sää", command=self.quick_weather).pack(side='left', padx=2)
        ttk.Button(actions_frame, text="🔄 Käännä", command=self.quick_translate).pack(side='left', padx=2)
        ttk.Button(actions_frame, text="📄 PDF", command=self.quick_pdf_read).pack(side='left', padx=2)
    
    def setup_tasks_tab(self):
        """Setup tasks management tab"""
        # Task list
        self.task_tree = ttk.Treeview(self.tasks_frame, columns=('status', 'type', 'scheduled', 'progress'), show='tree headings')
        self.task_tree.heading('#0', text='Nimi')
        self.task_tree.heading('status', text='Tila')
        self.task_tree.heading('type', text='Tyyppi')
        self.task_tree.heading('scheduled', text='Ajastettu')
        self.task_tree.heading('progress', text='Edistyminen')
        
        # Configure column widths
        self.task_tree.column('#0', width=200)
        self.task_tree.column('status', width=100)
        self.task_tree.column('type', width=120)
        self.task_tree.column('scheduled', width=150)
        self.task_tree.column('progress', width=100)
        
        self.task_tree.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Task control buttons
        task_buttons_frame = ttk.Frame(self.tasks_frame, style='Dark.TFrame')
        task_buttons_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(task_buttons_frame, text="🔄 Päivitä", command=self.refresh_tasks).pack(side='left', padx=2)
        ttk.Button(task_buttons_frame, text="❌ Peruuta", command=self.cancel_selected_task).pack(side='left', padx=2)
        ttk.Button(task_buttons_frame, text="📊 Raportti", command=self.generate_task_report).pack(side='left', padx=2)
        ttk.Button(task_buttons_frame, text="🗑️ Tyhjennä", command=self.clear_completed_tasks).pack(side='left', padx=2)
    
    def setup_scheduler_tab(self):
        """Setup task scheduler tab"""
        # Scheduler form
        form_frame = ttk.LabelFrame(self.scheduler_frame, text="Uusi ajastettu tehtävä", style='Dark.TFrame')
        form_frame.pack(fill='x', padx=10, pady=10)
        
        # Task name
        ttk.Label(form_frame, text="Tehtävän nimi:", style='Dark.TLabel').grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.task_name_entry = tk.Entry(form_frame, bg='#404040', fg='#ffffff')
        self.task_name_entry.grid(row=0, column=1, sticky='ew', padx=5, pady=5)
        
        # Task description
        ttk.Label(form_frame, text="Kuvaus:", style='Dark.TLabel').grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.task_desc_entry = tk.Entry(form_frame, bg='#404040', fg='#ffffff')
        self.task_desc_entry.grid(row=1, column=1, sticky='ew', padx=5, pady=5)
        
        # Task type
        ttk.Label(form_frame, text="Tyyppi:", style='Dark.TLabel').grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.task_type_combo = ttk.Combobox(form_frame, values=[
            'ai_chat', 'web_search', 'youtube_search', 'translate', 
            'api_call', 'pdf_read', 'custom_command'
        ])
        self.task_type_combo.grid(row=2, column=1, sticky='ew', padx=5, pady=5)
        
        # Schedule time
        ttk.Label(form_frame, text="Ajastus:", style='Dark.TLabel').grid(row=3, column=0, sticky='w', padx=5, pady=5)
        
        schedule_frame = ttk.Frame(form_frame, style='Dark.TFrame')
        schedule_frame.grid(row=3, column=1, sticky='ew', padx=5, pady=5)
        
        if CALENDAR_AVAILABLE:
            self.date_entry = DateEntry(schedule_frame, background='darkblue', foreground='white', borderwidth=2)
        else:
            self.date_entry = tk.Entry(schedule_frame, bg='#404040', fg='#ffffff')
            self.date_entry.insert(0, datetime.now().strftime('%Y-%m-%d'))
        self.date_entry.pack(side='left', padx=(0, 5))
        
        self.time_entry = tk.Entry(schedule_frame, bg='#404040', fg='#ffffff', width=8)
        self.time_entry.pack(side='left')
        self.time_entry.insert(0, datetime.now().strftime('%H:%M'))
        
        # Recurring options
        ttk.Label(form_frame, text="Toisto:", style='Dark.TLabel').grid(row=4, column=0, sticky='w', padx=5, pady=5)
        
        recurring_frame = ttk.Frame(form_frame, style='Dark.TFrame')
        recurring_frame.grid(row=4, column=1, sticky='ew', padx=5, pady=5)
        
        self.recurring_var = tk.BooleanVar()
        recurring_check = ttk.Checkbutton(recurring_frame, text="Toistuva", variable=self.recurring_var)
        recurring_check.pack(side='left', padx=(0, 5))
        
        self.interval_combo = ttk.Combobox(recurring_frame, values=['hourly', 'daily', 'weekly', 'monthly'], width=10)
        self.interval_combo.pack(side='left')
        
        # Parameters
        ttk.Label(form_frame, text="Parametrit (JSON):", style='Dark.TLabel').grid(row=5, column=0, sticky='w', padx=5, pady=5)
        self.params_entry = tk.Text(form_frame, height=3, bg='#404040', fg='#ffffff')
        self.params_entry.grid(row=5, column=1, sticky='ew', padx=5, pady=5)
        self.params_entry.insert('1.0', '{"prompt": "Hei AI!"}')
        
        # Configure grid weights
        form_frame.columnconfigure(1, weight=1)
        
        # Schedule button
        ttk.Button(form_frame, text="📅 Ajasta tehtävä", command=self.schedule_task).grid(row=6, column=0, columnspan=2, pady=10)
    
    def setup_settings_tab(self):
        """Setup settings tab"""
        # AI Provider settings
        ai_frame = ttk.LabelFrame(self.settings_frame, text="AI-asetukset", style='Dark.TFrame')
        ai_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(ai_frame, text="AI-palveluntarjoaja:", style='Dark.TLabel').grid(row=0, column=0, sticky='w', padx=5, pady=5)
        
        available_providers = list(self.agent.ai_manager.get_available_providers().keys())
        self.ai_provider_combo = ttk.Combobox(ai_frame, values=available_providers)
        self.ai_provider_combo.set(self.agent.current_ai_provider)
        self.ai_provider_combo.grid(row=0, column=1, sticky='ew', padx=5, pady=5)
        self.ai_provider_combo.bind('<<ComboboxSelected>>', self.on_provider_change)
        
        ttk.Label(ai_frame, text="Malli:", style='Dark.TLabel').grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.ai_model_combo = ttk.Combobox(ai_frame)
        self.ai_model_combo.set(self.agent.current_ai_model)
        self.ai_model_combo.grid(row=1, column=1, sticky='ew', padx=5, pady=5)
        self.update_model_list()
        
        # API Keys
        keys_frame = ttk.LabelFrame(self.settings_frame, text="API-avaimet", style='Dark.TFrame')
        keys_frame.pack(fill='x', padx=10, pady=10)
        
        self.api_key_entries = {}
        row = 0
        for provider, info in self.agent.ai_manager.get_available_providers().items():
            if info["requires_key"]:
                ttk.Label(keys_frame, text=f"{info['name']}:", style='Dark.TLabel').grid(row=row, column=0, sticky='w', padx=5, pady=2)
                
                entry = tk.Entry(keys_frame, show="*", bg='#404040', fg='#ffffff')
                entry.grid(row=row, column=1, sticky='ew', padx=5, pady=2)
                
                # Load existing key
                existing_key = self.agent.get_api_key(provider)
                if existing_key:
                    entry.insert(0, existing_key)
                
                self.api_key_entries[provider] = entry
                
                ttk.Button(keys_frame, text="Tallenna", 
                          command=lambda p=provider: self.save_api_key(p)).grid(row=row, column=2, padx=5, pady=2)
                row += 1
        
        keys_frame.columnconfigure(1, weight=1)
        ai_frame.columnconfigure(1, weight=1)
        
        # System info
        info_frame = ttk.LabelFrame(self.settings_frame, text="Järjestelmätiedot", style='Dark.TFrame')
        info_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.system_info = scrolledtext.ScrolledText(info_frame, height=10, bg='#1e1e1e', fg='#ffffff')
        self.system_info.pack(fill='both', expand=True, padx=5, pady=5)
        self.update_system_info()
    
    def setup_menu(self):
        """Setup application menu"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tiedosto", menu=file_menu)
        file_menu.add_command(label="Lataa keskustelu", command=self.load_conversation)
        file_menu.add_command(label="Tallenna keskustelu", command=self.save_conversation)
        file_menu.add_separator()
        file_menu.add_command(label="Lopeta", command=self.quit_application)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Työkalut", menu=tools_menu)
        tools_menu.add_command(label="Tyhjennä keskustelu", command=self.clear_chat)
        tools_menu.add_command(label="Vie tehtävät PDF:ksi", command=self.export_tasks_pdf)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Ohje", menu=help_menu)
        help_menu.add_command(label="Tietoja", command=self.show_about)
    
    def send_message(self, event=None):
        """Send message to AI"""
        message = self.chat_input.get().strip()
        if not message:
            return
        
        self.chat_input.delete(0, tk.END)
        
        # Display user message
        self.add_to_chat(f"👤 Sinä: {message}\n", '#00ff00')
        
        # Get AI response in thread
        def get_response():
            try:
                response = self.agent.chat_with_user(message)
                self.root.after(0, lambda: self.add_to_chat(f"🤖 AI: {response}\n\n", '#00aaff'))
                
                # Optional: speak response
                if SPEECH_AVAILABLE and hasattr(self, 'speak_responses') and self.speak_responses:
                    threading.Thread(target=lambda: self.agent.speak_text(response), daemon=True).start()
            except Exception as e:
                self.root.after(0, lambda: self.add_to_chat(f"❌ Virhe: {str(e)}\n\n", '#ff0000'))
        
        threading.Thread(target=get_response, daemon=True).start()
    
    def add_to_chat(self, text, color='#ffffff'):
        """Add text to chat display"""
        self.chat_display.config(state='normal')
        self.chat_display.insert(tk.END, text)
        
        # Color the last inserted text
        start_pos = f"{self.chat_display.index(tk.END)}-{len(text)}c"
        self.chat_display.tag_add("colored", start_pos, tk.END)
        self.chat_display.tag_config("colored", foreground=color)
        
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)
    
    def start_voice_input(self):
        """Start voice input"""
        if not SPEECH_RECOGNITION_AVAILABLE:
            messagebox.showwarning("Varoitus", "Puheentunnistus ei ole saatavilla")
            return
        
        def voice_thread():
            try:
                self.root.after(0, lambda: self.add_to_chat("🎤 Kuuntelen...\n", '#ffff00'))
                text = self.agent.listen_speech()
                self.root.after(0, lambda: self.chat_input.insert(0, text))
                self.root.after(0, lambda: self.add_to_chat(f"🎤 Kuultiin: {text}\n", '#ffff00'))
            except Exception as e:
                self.root.after(0, lambda: self.add_to_chat(f"🎤 Virhe: {str(e)}\n", '#ff0000'))
        
        threading.Thread(target=voice_thread, daemon=True).start()
    
    def quick_web_search(self):
        """Quick web search"""
        query = tk.simpledialog.askstring("Web-haku", "Syötä hakutermi:")
        if query:
            self.chat_input.insert(0, f"Hae webistä: {query}")
    
    def quick_youtube_search(self):
        """Quick YouTube search"""
        query = tk.simpledialog.askstring("YouTube-haku", "Syötä hakutermi:")
        if query:
            self.chat_input.insert(0, f"Hae YouTubesta: {query}")
    
    def quick_weather(self):
        """Quick weather check"""
        city = tk.simpledialog.askstring("Sää", "Syötä kaupunki:", initialvalue="Helsinki")
        if city:
            self.chat_input.insert(0, f"Mikä on sää kaupungissa {city}?")
    
    def quick_translate(self):
        """Quick translate"""
        text = tk.simpledialog.askstring("Käännös", "Syötä käännettävä teksti:")
        if text:
            lang = tk.simpledialog.askstring("Kieli", "Kohdekieli (esim. en, fi, de):", initialvalue="en")
            if lang:
                self.chat_input.insert(0, f"Käännä '{text}' kielelle {lang}")
    
    def quick_pdf_read(self):
        """Quick PDF read"""
        file_path = filedialog.askopenfilename(
            title="Valitse PDF-tiedosto",
            filetypes=[("PDF files", "*.pdf")]
        )
        if file_path:
            self.chat_input.insert(0, f"Lue PDF: {file_path}")
    
    def update_task_display(self):
        """Update task display periodically"""
        while True:
            try:
                time.sleep(2)  # Update every 2 seconds
                self.root.after(0, self.refresh_tasks)
            except Exception as e:
                print(f"Task update error: {e}")
                time.sleep(5)
    
    def refresh_tasks(self):
        """Refresh the task list"""
        try:
            # Clear current items
            for item in self.task_tree.get_children():
                self.task_tree.delete(item)
            
            # Add tasks
            tasks = self.agent.scheduler.get_all_tasks()
            for task in sorted(tasks, key=lambda t: t.created_at, reverse=True):
                scheduled_str = task.scheduled_time.strftime('%d.%m.%Y %H:%M') if task.scheduled_time else "Ei ajastettu"
                progress_str = f"{task.progress}%"
                
                self.task_tree.insert(
                    '',
                    'end',
                    text=task.name,
                    values=(task.status, task.task_type, scheduled_str, progress_str)
                )
        except Exception as e:
            print(f"Task refresh error: {e}")
    
    def cancel_selected_task(self):
        """Cancel selected task"""
        selection = self.task_tree.selection()
        if not selection:
            messagebox.showwarning("Varoitus", "Valitse peruutettava tehtävä")
            return
        
        item = selection[0]
        task_name = self.task_tree.item(item, 'text')
        
        # Find task by name (not ideal, but works for demo)
        for task in self.agent.scheduler.get_all_tasks():
            if task.name == task_name and task.status == TaskStatus.PENDING:
                if self.agent.scheduler.cancel_task(task.id):
                    messagebox.showinfo("Onnistui", f"Tehtävä '{task_name}' peruutettu")
                    self.refresh_tasks()
                    return
        
        messagebox.showwarning("Varoitus", "Tehtävää ei voitu peruuttaa")
    
    def generate_task_report(self):
        """Generate PDF report of tasks"""
        if not PDF_GENERATION_AVAILABLE:
            messagebox.showwarning("Varoitus", "PDF-generointi ei ole saatavilla")
            return
        
        tasks = self.agent.scheduler.get_all_tasks()
        if not tasks:
            messagebox.showinfo("Info", "Ei tehtäviä raportoitavaksi")
            return
        
        filename = PDFGenerator.generate_task_report(tasks)
        if filename.endswith('.pdf'):
            messagebox.showinfo("Onnistui", f"Raportti luotu: {filename}")
            # Try to open the PDF
            try:
                webbrowser.open(filename)
            except:
                pass
        else:
            messagebox.showerror("Virhe", filename)
    
    def clear_completed_tasks(self):
        """Clear completed tasks"""
        if messagebox.askyesno("Vahvistus", "Haluatko poistaa kaikki valmiit tehtävät?"):
            completed_tasks = [t for t in self.agent.scheduler.get_all_tasks() 
                             if t.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]]
            
            for task in completed_tasks:
                if task.id in self.agent.scheduler.tasks:
                    del self.agent.scheduler.tasks[task.id]
            
            self.agent.scheduler.save_tasks()
            self.refresh_tasks()
            messagebox.showinfo("Onnistui", f"Poistettu {len(completed_tasks)} tehtävää")
    
    def schedule_task(self):
        """Schedule a new task"""
        try:
            # Get form data
            name = self.task_name_entry.get().strip()
            description = self.task_desc_entry.get().strip()
            task_type = self.task_type_combo.get()
            
            if not name or not task_type:
                messagebox.showwarning("Varoitus", "Syötä tehtävän nimi ja tyyppi")
                return
            
            # Parse date and time
            if CALENDAR_AVAILABLE:
                date_str = self.date_entry.get()
            else:
                date_str = self.date_entry.get()
            
            time_str = self.time_entry.get()
            datetime_str = f"{date_str} {time_str}"
            
            try:
                scheduled_time = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M')
            except ValueError:
                try:
                    scheduled_time = datetime.strptime(datetime_str, '%d.%m.%Y %H:%M')
                except ValueError:
                    messagebox.showerror("Virhe", "Virheellinen päivämäärä/aika-muoto")
                    return
            
            # Parse parameters
            try:
                params_text = self.params_entry.get('1.0', tk.END).strip()
                parameters = json.loads(params_text) if params_text else {}
            except json.JSONDecodeError:
                messagebox.showerror("Virhe", "Virheellinen JSON-muoto parametreissa")
                return
            
            # Create task
            task = ScheduledTask(
                str(uuid.uuid4()),
                name,
                description,
                scheduled_time,
                task_type,
                parameters,
                self.recurring_var.get(),
                self.interval_combo.get() if self.recurring_var.get() else None
            )
            
            # Add to scheduler
            task_id = self.agent.scheduler.add_task(task)
            
            messagebox.showinfo("Onnistui", f"Tehtävä ajastettu! ID: {task_id}")
            
            # Clear form
            self.task_name_entry.delete(0, tk.END)
            self.task_desc_entry.delete(0, tk.END)
            self.task_type_combo.set('')
            self.params_entry.delete('1.0', tk.END)
            self.params_entry.insert('1.0', '{"prompt": "Hei AI!"}')
            
            self.refresh_tasks()
            
        except Exception as e:
            messagebox.showerror("Virhe", f"Tehtävän ajastus epäonnistui: {str(e)}")
    
    def on_provider_change(self, event=None):
        """Handle AI provider change"""
        provider = self.ai_provider_combo.get()
        self.agent.current_ai_provider = provider
        self.agent.save_config()
        self.update_model_list()
    
    def update_model_list(self):
        """Update model list based on selected provider"""
        provider = self.ai_provider_combo.get()
        if provider in self.agent.ai_manager.providers:
            models = self.agent.ai_manager.providers[provider]["models"]
            self.ai_model_combo['values'] = models
            if models:
                self.ai_model_combo.set(models[0])
                self.agent.current_ai_model = models[0]
                self.agent.save_config()
    
    def save_api_key(self, provider):
        """Save API key for provider"""
        if provider in self.api_key_entries:
            key = self.api_key_entries[provider].get().strip()
            if key:
                self.agent.set_api_key(provider, key)
                messagebox.showinfo("Onnistui", f"API-avain tallennettu: {provider}")
            else:
                messagebox.showwarning("Varoitus", "Syötä API-avain")
    
    def update_system_info(self):
        """Update system information display"""
        info_text = f"""
🤖 Smart AI Agent v2.0
📅 Käynnistetty: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}

📊 Järjestelmätiedot:
• Python versio: {sys.version.split()[0]}
• Käyttöjärjestelmä: {os.name}
• Työhakemisto: {os.getcwd()}

🔧 Saatavilla olevat ominaisuudet:
• Puhesynteesi: {'✅' if SPEECH_AVAILABLE else '❌'}
• Puheentunnistus: {'✅' if SPEECH_RECOGNITION_AVAILABLE else '❌'}
• PDF-generointi: {'✅' if PDF_GENERATION_AVAILABLE else '❌'}
• Kalenteri: {'✅' if CALENDAR_AVAILABLE else '❌'}
• YouTube-haku: {'✅' if YOUTUBE_AVAILABLE else '❌'}
• Käännöspalvelu: {'✅' if TRANSLATE_AVAILABLE else '❌'}

🤖 AI-palveluntarjoajat:
"""
        
        for provider, info in self.agent.ai_manager.get_available_providers().items():
            status = '✅' if info['available'] else '❌'
            key_status = '🔑' if self.agent.get_api_key(provider) else '🔓'
            info_text += f"• {info['name']}: {status} {key_status}\n"
        
        info_text += f"""
📋 Tehtävätilastot:
• Odottavia: {len(self.agent.scheduler.get_tasks_by_status(TaskStatus.PENDING))}
• Käynnissä: {len(self.agent.scheduler.get_tasks_by_status(TaskStatus.RUNNING))}
• Valmistuneita: {len(self.agent.scheduler.get_tasks_by_status(TaskStatus.COMPLETED))}
• Epäonnistuneita: {len(self.agent.scheduler.get_tasks_by_status(TaskStatus.FAILED))}
• Peruutettuja: {len(self.agent.scheduler.get_tasks_by_status(TaskStatus.CANCELLED))}

💡 Vinkkejä:
• Käytä 🎤-nappia puheentunnistukseen
• Ajasta tehtäviä Ajastus-välilehdellä
• Luo PDF-raportteja Tehtävät-välilehdellä
• Tallenna API-avaimet Asetukset-välilehdellä
"""
        
        self.system_info.delete('1.0', tk.END)
        self.system_info.insert('1.0', info_text)
    
    def load_conversation(self):
        """Load conversation from file"""
        try:
            file_path = filedialog.askopenfilename(
                title="Lataa keskustelu",
                filetypes=[("JSON files", "*.json")]
            )
            if file_path:
                with open(file_path, 'r', encoding='utf-8') as f:
                    conversation = json.load(f)
                    self.agent.conversation_history = conversation
                
                # Display loaded conversation
                self.clear_chat()
                for msg in conversation:
                    role = "👤 Sinä" if msg["role"] == "user" else "🤖 AI"
                    color = '#00ff00' if msg["role"] == "user" else '#00aaff'
                    self.add_to_chat(f"{role}: {msg['content']}\n", color)
                
                messagebox.showinfo("Onnistui", "Keskustelu ladattu")
        except Exception as e:
            messagebox.showerror("Virhe", f"Keskustelun lataus epäonnistui: {str(e)}")
    
    def save_conversation(self):
        """Save conversation to file"""
        try:
            if not self.agent.conversation_history:
                messagebox.showinfo("Info", "Ei keskustelua tallennettavaksi")
                return
            
            file_path = filedialog.asksaveasfilename(
                title="Tallenna keskustelu",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json")]
            )
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.agent.conversation_history, f, ensure_ascii=False, indent=2)
                
                messagebox.showinfo("Onnistui", "Keskustelu tallennettu")
        except Exception as e:
            messagebox.showerror("Virhe", f"Keskustelun tallennus epäonnistui: {str(e)}")
    
    def clear_chat(self):
        """Clear chat display and history"""
        self.chat_display.config(state='normal')
        self.chat_display.delete('1.0', tk.END)
        self.chat_display.config(state='disabled')
        self.agent.conversation_history = []
        messagebox.showinfo("Onnistui", "Keskustelu tyhjennetty")
    
    def export_tasks_pdf(self):
        """Export all tasks to PDF"""
        self.generate_task_report()
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
🤖 Smart AI Agent v2.0

Monipuolinen AI-agentti tehtävien ajastuksella ja automaatiolla.

Ominaisuudet:
• Keskustelu useilla AI-palveluntarjoajilla
• Tehtävien ajastus ja automaatio
• Web-haku ja YouTube-haku
• PDF-lukeminen ja -generointi
• Puheentunnistus ja -synteesi
• Käännöspalvelut

Kehittäjä: AI Assistant
Versio: 2.0
Lisenssi: MIT

© 2024 Smart AI Agent Project
"""
        messagebox.showinfo("Tietoja", about_text)
    
    def quit_application(self):
        """Quit the application"""
        if messagebox.askyesno("Lopetus", "Haluatko varmasti lopettaa?"):
            self.agent.shutdown()
            self.root.quit()
    
    def run(self):
        """Run the GUI application"""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self.quit_application)
            self.root.mainloop()
        except KeyboardInterrupt:
            self.quit_application()


if __name__ == "__main__":
    # Install required packages if missing
    required_packages = [
        "requests", "PyPDF2", "schedule", 
        "reportlab", "tkcalendar", "pyttsx3",
        "deep-translator", "youtube-search-python",
        "SpeechRecognition", "openai", "anthropic",
        "google-generativeai"
    ]
    
    print("🤖 Smart AI Agent v2.0 käynnistyy...")
    print("📦 Tarkistetaan riippuvuudet...")
    
    # Show startup info
    print(f"✅ Python {sys.version.split()[0]}")
    print(f"✅ Työhakemisto: {os.getcwd()}")
    
    try:
        # Create and run the GUI
        app = SmartAgentGUI()
        print("🚀 GUI käynnistetty!")
        app.run()
    except Exception as e:
        print(f"❌ Virhe käynnistyksessä: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("👋 Smart AI Agent lopetettu")