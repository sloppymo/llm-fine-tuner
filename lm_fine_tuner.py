#!/usr/bin/env python3
# SylvaFine - Advanced Language Model Fine-Tuning Tool
# Created by Morgan Roberts MSW
# Version 1.1.0

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import json
import jsonlines
import threading
import os
import csv
import uuid
import re
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import openai
import time
import webbrowser
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.metrics.distance import edit_distance
from dotenv import load_dotenv

# Initialize NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Load environment variables
def load_environment():
    try:
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
            
        print(f"OpenAI API key loaded successfully: {api_key[:5]}... (truncated for security)")
        return api_key
        
    except Exception as e:
        print(f"Error loading environment: {str(e)}")
        return None

# Configure OpenAI
api_key = load_environment()
if not api_key:
    print("Failed to load environment variables. Exiting...")
    exit(1)

# Initialize OpenAI client with explicit parameters
client = openai.OpenAI(
    api_key=api_key,
    base_url="https://api.openai.com/v1"
)

class SylvaFineGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SylvaFine - Advanced LM Fine-Tuning Tool")
        self.root.geometry("1000x750")
        self.root.minsize(900, 650)
        
        # Set app icon if available
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass
            
        # Basic Variables
        self.dataset_path = tk.StringVar()
        self.epochs = tk.StringVar(value="3")
        self.batch_size = tk.StringVar(value="4")
        self.learning_rate = tk.StringVar(value="5e-5")
        self.test_prompt = tk.StringVar()
        self.selected_model = tk.StringVar(value="gpt-3.5-turbo")
        self.theme_var = tk.StringVar(value="Default")
        
        # Fine-tuning Variables
        self.suffix = tk.StringVar(value="")
        self.validation_split = tk.StringVar(value="0.1")
        self.hyperparams = {
            "n_epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "validation_split": self.validation_split
        }
        self.fine_tune_job_id = tk.StringVar()
        self.fine_tuned_model_id = tk.StringVar()
        
        # Dataset Editor Variables
        self.dataset_examples = []
        self.current_example_index = 0
        self.system_message = tk.StringVar(value="You are a helpful assistant.")
        self.user_message = tk.StringVar()
        self.assistant_message = tk.StringVar()
        self.dataset_modified = False
        
        # Evaluation Variables
        self.evaluation_metrics = {
            "bleu": [],
            "meteor": [],
            "edit_distance": [],
            "response_length": [],
            "response_time": []
        }
        self.test_dataset_path = tk.StringVar()
        self.test_results = []
        self.confusion_matrix_data = None
        self.evaluation_categories = []
        
        # State Variables
        self.training_active = False
        self.response_history = []
        self.training_metrics = {
            "loss": [], 
            "accuracy": [],
            "learning_rate": [],
            "batch_completion": [],
            "validation_loss": []
        }
        
        # Setup theme
        self.setup_theme()
        
        # Create main frames
        self.create_frames()
        self.create_widgets()
        
        # Show welcome message
        self.log_message("Welcome to SylvaFine - Advanced Language Model Fine-Tuning Tool v1.1.0")
        self.log_message("Created by Morgan Roberts MSW")
        self.log_message("Ready to start fine-tuning your language models!")
        
        # Initialize OpenAI fine-tuning job status checker
        self.job_status_check_id = None
    
    def setup_theme(self):
        # Create theme styles
        self.style = ttk.Style()
        
        # Default theme is already set
        # We'll add more theme options later
        pass
        
    def create_frames(self):
        # Main container with notebook for tabs
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Create menu
        self.create_menu()
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Main tab
        self.main_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.main_tab, text="Main")
        
        # Dataset Editor tab (new)
        self.dataset_editor_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.dataset_editor_tab, text="Dataset Editor")
        
        # Fine-Tuning tab (new)
        self.fine_tuning_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.fine_tuning_tab, text="Fine-Tuning")
        
        # Visualization tab
        self.viz_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.viz_tab, text="Visualization")
        
        # Evaluation tab (new)
        self.evaluation_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.evaluation_tab, text="Evaluation")
        
        # History tab
        self.history_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.history_tab, text="Response History")
        
        # Settings tab
        self.settings_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.settings_tab, text="Settings")
        
        # About tab
        self.about_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.about_tab, text="About")
        
        # Status frame in main tab
        self.status_frame = ttk.LabelFrame(self.main_tab, text="Status", padding="5")
        self.status_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        # Dataset frame in main tab
        self.dataset_frame = ttk.LabelFrame(self.main_tab, text="Dataset", padding="5")
        self.dataset_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Training settings frame in main tab
        self.training_frame = ttk.LabelFrame(self.main_tab, text="Training Settings", padding="5")
        self.training_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Test frame in main tab
        self.test_frame = ttk.LabelFrame(self.main_tab, text="Test Model", padding="5")
        self.test_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Configure grid weights for main tab
        self.main_tab.columnconfigure(0, weight=1)
        self.main_tab.rowconfigure(0, weight=1)
        self.main_tab.rowconfigure(1, weight=0)
        self.main_tab.rowconfigure(2, weight=0)
        self.main_tab.rowconfigure(3, weight=0)
        
        # Configure status frame
        self.status_frame.rowconfigure(0, weight=1)
        self.status_frame.columnconfigure(0, weight=1)
        
        # Configure main frame
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(0, weight=1)
        
        # Bind tab change event to handle tab-specific updates
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        
    def create_menu(self):
        # Create main menu
        self.menu_bar = tk.Menu(self.root)
        self.root.config(menu=self.menu_bar)
        
        # File menu
        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Project", command=self.new_project)
        file_menu.add_command(label="Open Project", command=self.open_project)
        file_menu.add_command(label="Save Project", command=self.save_project)
        file_menu.add_separator()
        file_menu.add_command(label="Export Responses", command=self.export_responses)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Tools menu
        tools_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Validate Dataset", command=self.validate_dataset)
        tools_menu.add_command(label="View Dataset Statistics", command=self.view_dataset_stats)
        tools_menu.add_separator()
        tools_menu.add_command(label="Clear Response History", command=self.clear_history)
        
        # Help menu
        help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Documentation", command=lambda: webbrowser.open("https://platform.openai.com/docs/guides/fine-tuning"))
        help_menu.add_command(label="About SylvaFine", command=self.show_about)
    
    def create_widgets(self):
        # Status window
        self.status_text = tk.Text(self.status_frame, height=10, width=80)
        self.status_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.status_scroll = ttk.Scrollbar(self.status_frame, orient=tk.VERTICAL, command=self.status_text.yview)
        self.status_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.status_text['yscrollcommand'] = self.status_scroll.set
        
        # Dataset upload widgets
        ttk.Label(self.dataset_frame, text="Dataset File:").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Entry(self.dataset_frame, textvariable=self.dataset_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(self.dataset_frame, text="Browse", command=self.browse_dataset).grid(row=0, column=2, padx=5)
        ttk.Button(self.dataset_frame, text="Analyze", command=self.view_dataset_stats).grid(row=0, column=3, padx=5)
        ttk.Button(self.dataset_frame, text="Edit", command=self.open_dataset_editor).grid(row=0, column=4, padx=5)
        
        # Training settings widgets
        model_frame = ttk.Frame(self.training_frame)
        model_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(model_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, padx=5)
        models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
        model_dropdown = ttk.Combobox(model_frame, textvariable=self.selected_model, values=models, width=15)
        model_dropdown.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(self.training_frame, text="Epochs:").grid(row=1, column=0, sticky=tk.W, padx=5)
        ttk.Entry(self.training_frame, textvariable=self.epochs, width=10).grid(row=1, column=1, padx=5)
        
        ttk.Label(self.training_frame, text="Batch Size:").grid(row=2, column=0, sticky=tk.W, padx=5)
        ttk.Entry(self.training_frame, textvariable=self.batch_size, width=10).grid(row=2, column=1, padx=5)
        
        ttk.Label(self.training_frame, text="Learning Rate:").grid(row=3, column=0, sticky=tk.W, padx=5)
        ttk.Entry(self.training_frame, textvariable=self.learning_rate, width=10).grid(row=3, column=1, padx=5)
        
        button_frame = ttk.Frame(self.training_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Start Training", command=self.start_training).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Stop Training", command=self.stop_training).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Fine-Tune", command=lambda: self.notebook.select(2)).grid(row=0, column=2, padx=5)
        
        # Test model widgets
        ttk.Label(self.test_frame, text="Test Prompt:").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Entry(self.test_frame, textvariable=self.test_prompt, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(self.test_frame, text="Generate Response", command=self.generate_response).grid(row=0, column=2, padx=5)
        ttk.Button(self.test_frame, text="Save Response", command=self.save_response).grid(row=0, column=3, padx=5)
        ttk.Button(self.test_frame, text="Evaluate", command=lambda: self.notebook.select(4)).grid(row=0, column=4, padx=5)
        
        # Create dataset editor tab content
        self.create_dataset_editor_tab()
        
        # Create fine-tuning tab content
        self.create_fine_tuning_tab()
        
        # Create visualization tab content
        self.create_visualization_tab()
        
        # Create evaluation tab content
        self.create_evaluation_tab()
        
        # Create history tab content
        self.create_history_tab()
        
        # Create settings tab content
        self.create_settings_tab()
        
        # Create about tab content
        self.create_about_tab()
    
    def on_tab_changed(self, event):
        """Handle tab change events"""
        tab_id = self.notebook.index("current")
        
        # Update visualization if on visualization tab
        if tab_id == 3:  # Visualization tab
            self.update_plot()
        
        # Update dataset editor if on dataset editor tab
        elif tab_id == 1:  # Dataset Editor tab
            if self.dataset_path.get() and os.path.exists(self.dataset_path.get()) and not self.dataset_examples:
                self.load_dataset_for_editing()
        
        # Update fine-tuning status if on fine-tuning tab
        elif tab_id == 2:  # Fine-Tuning tab
            if self.fine_tune_job_id.get():
                self.check_fine_tune_job_status()
    
    def create_dataset_editor_tab(self):
        """Create the dataset editor tab content"""
        # Create main frames
        editor_frame = ttk.LabelFrame(self.dataset_editor_tab, text="Dataset Editor", padding="10")
        editor_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        
        # Dataset controls
        controls_frame = ttk.Frame(editor_frame, padding="5")
        controls_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(controls_frame, text="Dataset:").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Entry(controls_frame, textvariable=self.dataset_path, width=40, state="readonly").grid(row=0, column=1, padx=5)
        ttk.Button(controls_frame, text="Load", command=self.load_dataset_for_editing).grid(row=0, column=2, padx=5)
        ttk.Button(controls_frame, text="New", command=self.new_dataset).grid(row=0, column=3, padx=5)
        ttk.Button(controls_frame, text="Save", command=self.save_dataset).grid(row=0, column=4, padx=5)
        
        # Navigation controls
        nav_frame = ttk.Frame(editor_frame, padding="5")
        nav_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(nav_frame, text="« First", command=lambda: self.navigate_examples("first")).grid(row=0, column=0, padx=5)
        ttk.Button(nav_frame, text="‹ Previous", command=lambda: self.navigate_examples("prev")).grid(row=0, column=1, padx=5)
        self.example_counter_label = ttk.Label(nav_frame, text="Example 0 of 0")
        self.example_counter_label.grid(row=0, column=2, padx=20)
        ttk.Button(nav_frame, text="Next ›", command=lambda: self.navigate_examples("next")).grid(row=0, column=3, padx=5)
        ttk.Button(nav_frame, text="Last »", command=lambda: self.navigate_examples("last")).grid(row=0, column=4, padx=5)
        
        # Message editing
        message_frame = ttk.LabelFrame(editor_frame, text="Conversation", padding="10")
        message_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        message_frame.columnconfigure(1, weight=1)
        
        # System message
        ttk.Label(message_frame, text="System:").grid(row=0, column=0, sticky=tk.NW, padx=5, pady=5)
        self.system_text = tk.Text(message_frame, height=2, width=60, wrap=tk.WORD)
        self.system_text.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        system_scroll = ttk.Scrollbar(message_frame, orient=tk.VERTICAL, command=self.system_text.yview)
        system_scroll.grid(row=0, column=2, sticky=(tk.N, tk.S))
        self.system_text['yscrollcommand'] = system_scroll.set
        
        # User message
        ttk.Label(message_frame, text="User:").grid(row=1, column=0, sticky=tk.NW, padx=5, pady=5)
        self.user_text = tk.Text(message_frame, height=5, width=60, wrap=tk.WORD)
        self.user_text.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        user_scroll = ttk.Scrollbar(message_frame, orient=tk.VERTICAL, command=self.user_text.yview)
        user_scroll.grid(row=1, column=2, sticky=(tk.N, tk.S))
        self.user_text['yscrollcommand'] = user_scroll.set
        
        # Assistant message
        ttk.Label(message_frame, text="Assistant:").grid(row=2, column=0, sticky=tk.NW, padx=5, pady=5)
        self.assistant_text = tk.Text(message_frame, height=5, width=60, wrap=tk.WORD)
        self.assistant_text.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        assistant_scroll = ttk.Scrollbar(message_frame, orient=tk.VERTICAL, command=self.assistant_text.yview)
        assistant_scroll.grid(row=2, column=2, sticky=(tk.N, tk.S))
        self.assistant_text['yscrollcommand'] = assistant_scroll.set
        
        # Example management buttons
        example_buttons_frame = ttk.Frame(editor_frame, padding="5")
        example_buttons_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Button(example_buttons_frame, text="Add Example", command=self.add_example).grid(row=0, column=0, padx=5)
        ttk.Button(example_buttons_frame, text="Update Example", command=self.update_example).grid(row=0, column=1, padx=5)
        ttk.Button(example_buttons_frame, text="Delete Example", command=self.delete_example).grid(row=0, column=2, padx=5)
        ttk.Button(example_buttons_frame, text="Generate Assistant Response", command=self.generate_assistant_response).grid(row=0, column=3, padx=5)
        
        # Configure grid weights
        self.dataset_editor_tab.columnconfigure(0, weight=1)
        self.dataset_editor_tab.rowconfigure(0, weight=1)
        editor_frame.columnconfigure(0, weight=1)
        editor_frame.rowconfigure(2, weight=1)
    
    def open_dataset_editor(self):
        """Open the dataset editor tab"""
        if not self.dataset_path.get():
            messagebox.showerror("Error", "Please select a dataset file first")
            return
            
        # Switch to dataset editor tab
        self.notebook.select(1)  # Index 1 is the Dataset Editor tab
        
        # Load the dataset if not already loaded
        if not self.dataset_examples:
            self.load_dataset_for_editing()
    
    def new_dataset(self):
        """Create a new empty dataset"""
        # Ask for file location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".jsonl",
            filetypes=[("JSONL files", "*.jsonl"), ("All files", "*.*")],
            title="Create New Dataset"
        )
        
        if not file_path:
            return  # User cancelled
            
        # Set the dataset path
        self.dataset_path.set(file_path)
        
        # Clear any existing examples
        self.dataset_examples = []
        self.current_example_index = 0
        
        # Create an empty file
        with open(file_path, 'w') as f:
            pass  # Create empty file
            
        # Update UI
        self.update_example_display()
        self.log_message(f"Created new dataset: {file_path}")
        self.dataset_modified = False
    
    def load_dataset_for_editing(self):
        """Load the dataset for editing"""
        if not self.dataset_path.get() or not os.path.exists(self.dataset_path.get()):
            messagebox.showerror("Error", "Please select a valid dataset file first")
            return
            
        # Clear existing examples
        self.dataset_examples = []
        self.current_example_index = 0
        
        try:
            # Read the JSONL file
            with jsonlines.open(self.dataset_path.get(), 'r') as reader:
                for obj in reader:
                    self.dataset_examples.append(obj)
                    
            # Update the display
            if self.dataset_examples:
                self.update_example_display()
                self.log_message(f"Loaded {len(self.dataset_examples)} examples from dataset")
            else:
                self.log_message("Dataset is empty")
                
            self.dataset_modified = False
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
            self.log_message(f"Error loading dataset: {str(e)}")
    
    def save_dataset(self):
        """Save the dataset to file"""
        if not self.dataset_path.get():
            messagebox.showerror("Error", "No dataset file specified")
            return
            
        try:
            # Write all examples to the JSONL file
            with jsonlines.open(self.dataset_path.get(), 'w') as writer:
                for example in self.dataset_examples:
                    writer.write(example)
                    
            self.log_message(f"Saved {len(self.dataset_examples)} examples to dataset")
            self.dataset_modified = False
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save dataset: {str(e)}")
            self.log_message(f"Error saving dataset: {str(e)}")
    
    def navigate_examples(self, direction):
        """Navigate through dataset examples"""
        if not self.dataset_examples:
            return
            
        # Check for unsaved changes
        if self.check_unsaved_changes():
            return
            
        # Calculate new index based on direction
        if direction == "first":
            self.current_example_index = 0
        elif direction == "prev":
            self.current_example_index = max(0, self.current_example_index - 1)
        elif direction == "next":
            self.current_example_index = min(len(self.dataset_examples) - 1, self.current_example_index + 1)
        elif direction == "last":
            self.current_example_index = len(self.dataset_examples) - 1
            
        # Update the display
        self.update_example_display()
    
    def check_unsaved_changes(self):
        """Check for unsaved changes in the current example"""
        # If no examples or no modifications, no need to check
        if not self.dataset_examples or not self.dataset_modified:
            return False
            
        # Ask user if they want to save changes
        response = messagebox.askyesnocancel("Unsaved Changes", 
                                            "You have unsaved changes. Would you like to save them?")
        
        if response is None:  # Cancel
            return True
        elif response:  # Yes
            self.update_example()
            
        return False
    
    def update_example_display(self):
        """Update the display with the current example"""
        # Clear text fields
        self.system_text.delete(1.0, tk.END)
        self.user_text.delete(1.0, tk.END)
        self.assistant_text.delete(1.0, tk.END)
        
        # If no examples, update counter and return
        if not self.dataset_examples:
            self.example_counter_label.config(text="Example 0 of 0")
            return
            
        # Get the current example
        example = self.dataset_examples[self.current_example_index]
        
        # Update counter
        self.example_counter_label.config(
            text=f"Example {self.current_example_index + 1} of {len(self.dataset_examples)}")
        
        # Extract messages
        if "messages" in example:
            messages = example["messages"]
            
            for message in messages:
                role = message.get("role", "")
                content = message.get("content", "")
                
                if role == "system":
                    self.system_text.insert(tk.END, content)
                elif role == "user":
                    self.user_text.insert(tk.END, content)
                elif role == "assistant":
                    self.assistant_text.insert(tk.END, content)
    
    def add_example(self):
        """Add a new example to the dataset"""
        # Check for unsaved changes
        if self.check_unsaved_changes():
            return
            
        # Create a new example with empty messages
        new_example = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": ""},
                {"role": "assistant", "content": ""}
            ]
        }
        
        # Add to dataset
        self.dataset_examples.append(new_example)
        self.current_example_index = len(self.dataset_examples) - 1
        
        # Update display
        self.update_example_display()
        self.log_message("Added new example")
        self.dataset_modified = True
    
    def update_example(self):
        """Update the current example with the text field contents"""
        if not self.dataset_examples:
            messagebox.showerror("Error", "No examples to update")
            return
            
        # Get text from fields
        system_content = self.system_text.get(1.0, tk.END).strip()
        user_content = self.user_text.get(1.0, tk.END).strip()
        assistant_content = self.assistant_text.get(1.0, tk.END).strip()
        
        # Validate inputs
        if not user_content:
            messagebox.showerror("Error", "User message cannot be empty")
            return
            
        if not assistant_content:
            messagebox.showerror("Error", "Assistant message cannot be empty")
            return
            
        # Create updated example
        updated_example = {
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        }
        
        # Update dataset
        self.dataset_examples[self.current_example_index] = updated_example
        
        # Update UI
        self.log_message(f"Updated example {self.current_example_index + 1}")
        self.dataset_modified = True
    
    def delete_example(self):
        """Delete the current example"""
        if not self.dataset_examples:
            messagebox.showerror("Error", "No examples to delete")
            return
            
        # Confirm deletion
        if not messagebox.askyesno("Confirm Delete", "Are you sure you want to delete this example?"):
            return
            
        # Remove the example
        del self.dataset_examples[self.current_example_index]
        
        # Update index
        if self.dataset_examples:
            self.current_example_index = min(self.current_example_index, len(self.dataset_examples) - 1)
            self.update_example_display()
        else:
            self.current_example_index = 0
            self.update_example_display()  # Will show empty display
            
        self.log_message("Deleted example")
        self.dataset_modified = True
    
    def generate_assistant_response(self):
        """Generate an assistant response for the current example"""
        # Get system and user messages
        system_content = self.system_text.get(1.0, tk.END).strip()
        user_content = self.user_text.get(1.0, tk.END).strip()
        
        if not user_content:
            messagebox.showerror("Error", "User message cannot be empty")
            return
            
        try:
            # Create messages for API call
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
            
            # Call OpenAI API
            self.log_message("Generating assistant response...")
            response = client.chat.completions.create(
                model=self.selected_model.get(),
                messages=messages,
                temperature=0.7,
                max_tokens=1024
            )
            
            # Extract and display the response
            assistant_response = response.choices[0].message.content
            self.assistant_text.delete(1.0, tk.END)
            self.assistant_text.insert(tk.END, assistant_response)
            
            self.log_message("Generated assistant response")
            self.dataset_modified = True
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate response: {str(e)}")
            self.log_message(f"Error generating response: {str(e)}")
    
    def create_fine_tuning_tab(self):
        """Create the fine-tuning tab content"""
        # Create main frames
        fine_tuning_frame = ttk.LabelFrame(self.fine_tuning_tab, text="OpenAI Fine-Tuning", padding="10")
        fine_tuning_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        
        # Dataset selection
        dataset_frame = ttk.Frame(fine_tuning_frame, padding="5")
        dataset_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(dataset_frame, text="Dataset File:").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Entry(dataset_frame, textvariable=self.dataset_path, width=40, state="readonly").grid(row=0, column=1, padx=5)
        ttk.Button(dataset_frame, text="Browse", command=self.browse_dataset).grid(row=0, column=2, padx=5)
        ttk.Button(dataset_frame, text="Validate", command=self.validate_dataset).grid(row=0, column=3, padx=5)
        
        # Hyperparameters
        params_frame = ttk.LabelFrame(fine_tuning_frame, text="Hyperparameters", padding="10")
        params_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=10)
        
        # Base model
        ttk.Label(params_frame, text="Base Model:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
        model_dropdown = ttk.Combobox(params_frame, textvariable=self.selected_model, values=models, width=15)
        model_dropdown.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Model suffix
        ttk.Label(params_frame, text="Model Suffix:").grid(row=0, column=2, sticky=tk.W, padx=15, pady=5)
        ttk.Entry(params_frame, textvariable=self.suffix, width=15).grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)
        
        # Epochs
        ttk.Label(params_frame, text="Epochs:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(params_frame, textvariable=self.epochs, width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Batch size
        ttk.Label(params_frame, text="Batch Size:").grid(row=1, column=2, sticky=tk.W, padx=15, pady=5)
        ttk.Entry(params_frame, textvariable=self.batch_size, width=10).grid(row=1, column=3, sticky=tk.W, padx=5, pady=5)
        
        # Learning rate
        ttk.Label(params_frame, text="Learning Rate:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(params_frame, textvariable=self.learning_rate, width=10).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Validation split
        ttk.Label(params_frame, text="Validation Split:").grid(row=2, column=2, sticky=tk.W, padx=15, pady=5)
        ttk.Entry(params_frame, textvariable=self.validation_split, width=10).grid(row=2, column=3, sticky=tk.W, padx=5, pady=5)
        
        # Fine-tuning controls
        control_frame = ttk.Frame(fine_tuning_frame, padding="5")
        control_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Button(control_frame, text="Start Fine-Tuning", command=self.start_openai_fine_tuning).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="Check Status", command=self.check_fine_tune_job_status).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Cancel Job", command=self.cancel_fine_tune_job).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="List Jobs", command=self.list_fine_tune_jobs).grid(row=0, column=3, padx=5)
        
        # Job status
        status_frame = ttk.LabelFrame(fine_tuning_frame, text="Job Status", padding="10")
        status_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        # Job ID
        ttk.Label(status_frame, text="Job ID:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(status_frame, textvariable=self.fine_tune_job_id, width=30, state="readonly").grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Model ID
        ttk.Label(status_frame, text="Fine-Tuned Model:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(status_frame, textvariable=self.fine_tuned_model_id, width=30, state="readonly").grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Status text
        self.job_status_text = scrolledtext.ScrolledText(status_frame, height=10, width=70, wrap=tk.WORD)
        self.job_status_text.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        self.job_status_text.config(state=tk.DISABLED)
        
        # Configure grid weights
        self.fine_tuning_tab.columnconfigure(0, weight=1)
        self.fine_tuning_tab.rowconfigure(0, weight=1)
        fine_tuning_frame.columnconfigure(0, weight=1)
        fine_tuning_frame.rowconfigure(3, weight=1)
        status_frame.columnconfigure(1, weight=1)
        status_frame.rowconfigure(2, weight=1)
    
    def start_openai_fine_tuning(self):
        """Start a real OpenAI fine-tuning job"""
        # Validate dataset and API key
        if not self.validate_dataset():
            return
            
        if not self.validate_api_key():
            return
            
        # Validate hyperparameters
        try:
            epochs = int(self.epochs.get())
            batch_size = int(self.batch_size.get())
            learning_rate = float(self.learning_rate.get())
            validation_split = float(self.validation_split.get())
            
            if epochs <= 0 or batch_size <= 0 or learning_rate <= 0 or validation_split < 0 or validation_split >= 1:
                raise ValueError("Invalid hyperparameter values")
                
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid hyperparameters: {str(e)}")
            return
            
        # Confirm with user
        if not messagebox.askyesno("Confirm Fine-Tuning", 
                                 "This will start a real fine-tuning job with OpenAI and may incur costs. Continue?"):
            return
            
        try:
            # Upload the dataset file
            self.log_message("Uploading dataset file to OpenAI...")
            with open(self.dataset_path.get(), "rb") as file:
                upload_response = client.files.create(
                    file=file,
                    purpose="fine-tune"
                )
                
            file_id = upload_response.id
            self.log_message(f"Dataset uploaded successfully. File ID: {file_id}")
            
            # Create a suffix for the model if provided
            suffix = None
            if self.suffix.get().strip():
                suffix = self.suffix.get().strip()
                
            # Start fine-tuning job
            self.log_message("Starting fine-tuning job...")
            job_response = client.fine_tuning.jobs.create(
                training_file=file_id,
                model=self.selected_model.get(),
                suffix=suffix,
                hyperparameters={
                    "n_epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate_multiplier": learning_rate
                }
            )
            
            # Store job ID
            job_id = job_response.id
            self.fine_tune_job_id.set(job_id)
            
            # Update UI
            self.log_message(f"Fine-tuning job started. Job ID: {job_id}")
            self.update_job_status_text(f"Job created: {job_id}\nStatus: {job_response.status}\nModel: {job_response.model}")
            
            # Start periodic status checking
            self.start_job_status_checker()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start fine-tuning job: {str(e)}")
            self.log_message(f"Error starting fine-tuning job: {str(e)}")
    
    def check_fine_tune_job_status(self):
        """Check the status of the current fine-tuning job"""
        if not self.fine_tune_job_id.get():
            messagebox.showerror("Error", "No active fine-tuning job")
            return
            
        if not self.validate_api_key():
            return
            
        try:
            # Get job status
            job = client.fine_tuning.jobs.retrieve(self.fine_tune_job_id.get())
            
            # Update UI
            status_text = f"Job ID: {job.id}\nStatus: {job.status}\nModel: {job.model}"
            
            # Add more details if available
            if hasattr(job, 'fine_tuned_model') and job.fine_tuned_model:
                self.fine_tuned_model_id.set(job.fine_tuned_model)
                status_text += f"\nFine-tuned Model: {job.fine_tuned_model}"
                
            if hasattr(job, 'created_at'):
                created_time = datetime.fromtimestamp(job.created_at)
                status_text += f"\nCreated: {created_time.strftime('%Y-%m-%d %H:%M:%S')}"
                
            if hasattr(job, 'finished_at') and job.finished_at:
                finished_time = datetime.fromtimestamp(job.finished_at)
                status_text += f"\nFinished: {finished_time.strftime('%Y-%m-%d %H:%M:%S')}"
                
            # Check for training metrics
            if hasattr(job, 'training_metrics') and job.training_metrics:
                metrics = job.training_metrics
                status_text += "\n\nTraining Metrics:\n"
                
                if hasattr(metrics, 'train_loss'):
                    status_text += f"Training Loss: {metrics.train_loss:.4f}\n"
                    
                if hasattr(metrics, 'train_accuracy'):
                    status_text += f"Training Accuracy: {metrics.train_accuracy:.4f}\n"
                    
                if hasattr(metrics, 'validation_loss'):
                    status_text += f"Validation Loss: {metrics.validation_loss:.4f}\n"
                    
                if hasattr(metrics, 'validation_accuracy'):
                    status_text += f"Validation Accuracy: {metrics.validation_accuracy:.4f}"
            
            # Update status text
            self.update_job_status_text(status_text)
            self.log_message(f"Fine-tuning job status: {job.status}")
            
            # If job is completed, stop the status checker
            if job.status in ["succeeded", "failed", "cancelled"]:
                self.stop_job_status_checker()
                
                if job.status == "succeeded" and hasattr(job, 'fine_tuned_model'):
                    messagebox.showinfo("Fine-Tuning Complete", 
                                       f"Fine-tuning job completed successfully!\nModel ID: {job.fine_tuned_model}")
                elif job.status == "failed":
                    messagebox.showerror("Fine-Tuning Failed", 
                                       "Fine-tuning job failed. Check the status for details.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to check job status: {str(e)}")
            self.log_message(f"Error checking job status: {str(e)}")
    
    def cancel_fine_tune_job(self):
        """Cancel the current fine-tuning job"""
        if not self.fine_tune_job_id.get():
            messagebox.showerror("Error", "No active fine-tuning job")
            return
            
        if not self.validate_api_key():
            return
            
        # Confirm with user
        if not messagebox.askyesno("Confirm Cancellation", 
                                 "Are you sure you want to cancel the current fine-tuning job?"):
            return
            
        try:
            # Cancel the job
            client.fine_tuning.jobs.cancel(self.fine_tune_job_id.get())
            
            # Update UI
            self.log_message(f"Cancelled fine-tuning job: {self.fine_tune_job_id.get()}")
            
            # Check status to update UI
            self.check_fine_tune_job_status()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to cancel job: {str(e)}")
            self.log_message(f"Error cancelling job: {str(e)}")
    
    def list_fine_tune_jobs(self):
        """List all fine-tuning jobs"""
        if not self.validate_api_key():
            return
            
        try:
            # Get list of jobs
            jobs = client.fine_tuning.jobs.list(limit=10)
            
            # Create a dialog to display jobs
            jobs_window = tk.Toplevel(self.root)
            jobs_window.title("Fine-Tuning Jobs")
            jobs_window.geometry("800x500")
            
            # Create a frame for the jobs list
            frame = ttk.Frame(jobs_window, padding="10")
            frame.pack(fill=tk.BOTH, expand=True)
            
            # Create a treeview to display jobs
            columns = ("id", "model", "status", "created", "finished")
            tree = ttk.Treeview(frame, columns=columns, show="headings")
            
            # Define headings
            tree.heading("id", text="Job ID")
            tree.heading("model", text="Model")
            tree.heading("status", text="Status")
            tree.heading("created", text="Created")
            tree.heading("finished", text="Finished")
            
            # Define column widths
            tree.column("id", width=200)
            tree.column("model", width=150)
            tree.column("status", width=100)
            tree.column("created", width=150)
            tree.column("finished", width=150)
            
            # Add scrollbar
            scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
            tree.configure(yscrollcommand=scrollbar.set)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            # Add jobs to the treeview
            for job in jobs.data:
                created_time = datetime.fromtimestamp(job.created_at).strftime('%Y-%m-%d %H:%M:%S')
                finished_time = "N/A"
                if hasattr(job, 'finished_at') and job.finished_at:
                    finished_time = datetime.fromtimestamp(job.finished_at).strftime('%Y-%m-%d %H:%M:%S')
                    
                tree.insert("", tk.END, values=(job.id, job.model, job.status, created_time, finished_time))
                
            # Add a button to load the selected job
            def load_selected_job():
                selected_items = tree.selection()
                if not selected_items:
                    messagebox.showerror("Error", "Please select a job")
                    return
                    
                job_id = tree.item(selected_items[0], "values")[0]
                self.fine_tune_job_id.set(job_id)
                self.check_fine_tune_job_status()
                jobs_window.destroy()
                
            button_frame = ttk.Frame(jobs_window, padding="10")
            button_frame.pack(fill=tk.X)
            
            ttk.Button(button_frame, text="Load Selected Job", command=load_selected_job).pack(side=tk.RIGHT)
            ttk.Button(button_frame, text="Close", command=jobs_window.destroy).pack(side=tk.RIGHT, padx=10)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to list jobs: {str(e)}")
            self.log_message(f"Error listing jobs: {str(e)}")
    
    def update_job_status_text(self, text):
        """Update the job status text widget"""
        self.job_status_text.config(state=tk.NORMAL)
        self.job_status_text.delete(1.0, tk.END)
        self.job_status_text.insert(tk.END, text)
        self.job_status_text.config(state=tk.DISABLED)
    
    def start_job_status_checker(self):
        """Start periodic checking of job status"""
        # Stop any existing checker
        self.stop_job_status_checker()
        
        # Define the checker function
        def check_status():
            if self.fine_tune_job_id.get():
                self.check_fine_tune_job_status()
                # Schedule next check in 30 seconds
                self.job_status_check_id = self.root.after(30000, check_status)
                
        # Start the first check
        self.job_status_check_id = self.root.after(5000, check_status)
    
    def stop_job_status_checker(self):
        """Stop the periodic job status checker"""
        if self.job_status_check_id:
            self.root.after_cancel(self.job_status_check_id)
            self.job_status_check_id = None
    
    def create_evaluation_tab(self):
        """Create the model evaluation tab content"""
        # Create main frames
        eval_frame = ttk.LabelFrame(self.evaluation_tab, text="Model Evaluation", padding="10")
        eval_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        
        # Test dataset selection
        dataset_frame = ttk.Frame(eval_frame, padding="5")
        dataset_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(dataset_frame, text="Test Dataset:").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Entry(dataset_frame, textvariable=self.test_dataset_path, width=40).grid(row=0, column=1, padx=5)
        ttk.Button(dataset_frame, text="Browse", command=self.browse_test_dataset).grid(row=0, column=2, padx=5)
        
        # Model selection
        model_frame = ttk.Frame(eval_frame, padding="5")
        model_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(model_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, padx=5)
        models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
        # Add fine-tuned models if available
        if self.fine_tuned_model_id.get():
            models.append(self.fine_tuned_model_id.get())
        model_dropdown = ttk.Combobox(model_frame, textvariable=self.selected_model, values=models, width=30)
        model_dropdown.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        # Evaluation controls
        control_frame = ttk.Frame(eval_frame, padding="5")
        control_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Button(control_frame, text="Run Evaluation", command=self.run_evaluation).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="Single Test", command=self.run_single_test).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Export Results", command=self.export_evaluation_results).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="Clear Results", command=self.clear_evaluation_results).grid(row=0, column=3, padx=5)
        
        # Notebook for evaluation results
        eval_notebook = ttk.Notebook(eval_frame)
        eval_notebook.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        # Results tab
        results_tab = ttk.Frame(eval_notebook, padding="10")
        eval_notebook.add(results_tab, text="Results Table")
        
        # Create results table
        self.results_text = scrolledtext.ScrolledText(results_tab, height=15, width=80, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        self.results_text.config(state=tk.DISABLED)
        
        # Metrics tab
        metrics_tab = ttk.Frame(eval_notebook, padding="10")
        eval_notebook.add(metrics_tab, text="Metrics")
        
        # Create metrics visualization
        metrics_frame = ttk.Frame(metrics_tab)
        metrics_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a figure for metrics plotting
        self.metrics_figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.metrics_canvas = FigureCanvasTkAgg(self.metrics_figure, metrics_frame)
        self.metrics_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Confusion Matrix tab
        confusion_tab = ttk.Frame(eval_notebook, padding="10")
        eval_notebook.add(confusion_tab, text="Confusion Matrix")
        
        # Create confusion matrix visualization
        confusion_frame = ttk.Frame(confusion_tab)
        confusion_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a figure for confusion matrix
        self.confusion_figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.confusion_canvas = FigureCanvasTkAgg(self.confusion_figure, confusion_frame)
        self.confusion_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights
        self.evaluation_tab.columnconfigure(0, weight=1)
        self.evaluation_tab.rowconfigure(0, weight=1)
        eval_frame.columnconfigure(0, weight=1)
        eval_frame.rowconfigure(3, weight=1)
    
    def create_visualization_tab(self):
        # Create a frame for the visualization
        viz_frame = ttk.LabelFrame(self.viz_tab, text="Training Metrics", padding="10")
        viz_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        
        # Create a notebook for different visualizations
        viz_notebook = ttk.Notebook(viz_frame)
        viz_notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Loss tab
        loss_tab = ttk.Frame(viz_notebook, padding="10")
        viz_notebook.add(loss_tab, text="Loss")
        
        # Create a figure for loss plotting
        self.loss_figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.loss_canvas = FigureCanvasTkAgg(self.loss_figure, loss_tab)
        self.loss_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Accuracy tab
        accuracy_tab = ttk.Frame(viz_notebook, padding="10")
        viz_notebook.add(accuracy_tab, text="Accuracy")
        
        # Create a figure for accuracy plotting
        self.accuracy_figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.accuracy_canvas = FigureCanvasTkAgg(self.accuracy_figure, accuracy_tab)
        self.accuracy_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Learning Rate tab
        lr_tab = ttk.Frame(viz_notebook, padding="10")
        viz_notebook.add(lr_tab, text="Learning Rate")
        
        # Create a figure for learning rate plotting
        self.lr_figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.lr_canvas = FigureCanvasTkAgg(self.lr_figure, lr_tab)
        self.lr_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Validation tab
        validation_tab = ttk.Frame(viz_notebook, padding="10")
        viz_notebook.add(validation_tab, text="Validation")
        
        # Create a figure for validation plotting
        self.validation_figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.validation_canvas = FigureCanvasTkAgg(self.validation_figure, validation_tab)
        self.validation_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Controls frame
        controls_frame = ttk.Frame(viz_frame, padding="5")
        controls_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=10)
        
        # Add export button
        ttk.Button(controls_frame, text="Export Plot", command=self.export_plot).grid(row=0, column=0, padx=5)
        ttk.Button(controls_frame, text="Refresh", command=self.update_plot).grid(row=0, column=1, padx=5)
        ttk.Button(controls_frame, text="Clear", command=self.clear_plots).grid(row=0, column=2, padx=5)
        
        # Configure grid weights
        self.viz_tab.columnconfigure(0, weight=1)
        self.viz_tab.rowconfigure(0, weight=1)
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.rowconfigure(0, weight=1)
    
    def create_history_tab(self):
        # Create a frame for the response history
        history_frame = ttk.LabelFrame(self.history_tab, text="Response History", padding="10")
        history_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        
        # Create a text widget to display the history
        self.history_text = scrolledtext.ScrolledText(history_frame, wrap=tk.WORD, width=80, height=20)
        self.history_text.pack(fill=tk.BOTH, expand=True)
        self.history_text.config(state=tk.DISABLED)
        
        # Add controls for history
        control_frame = ttk.Frame(self.history_tab, padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=10, pady=10)
        
        ttk.Button(control_frame, text="Clear History", command=self.clear_history).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Export History", command=self.export_responses).pack(side=tk.LEFT, padx=5)
        
        # Configure grid weights
        self.history_tab.columnconfigure(0, weight=1)
        self.history_tab.rowconfigure(0, weight=1)
    
    def create_settings_tab(self):
        # Create a frame for the settings
        settings_frame = ttk.LabelFrame(self.settings_tab, text="Application Settings", padding="10")
        settings_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        
        # Theme selection
        ttk.Label(settings_frame, text="Theme:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        themes = ["Default", "Light", "Dark"]
        theme_dropdown = ttk.Combobox(settings_frame, textvariable=self.theme_var, values=themes, width=15)
        theme_dropdown.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        theme_dropdown.bind("<<ComboboxSelected>>", self.change_theme)
        
        # API settings
        api_frame = ttk.LabelFrame(settings_frame, text="API Settings", padding="10")
        api_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=10)
        
        ttk.Label(api_frame, text="API Key:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Button(api_frame, text="Update API Key", command=self.update_api_key).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Save settings button
        ttk.Button(settings_frame, text="Save Settings", command=self.save_settings).grid(row=2, column=0, columnspan=2, pady=10)
        
        # Configure grid weights
        self.settings_tab.columnconfigure(0, weight=1)
        self.settings_tab.rowconfigure(0, weight=1)
    
    def create_about_tab(self):
        # Create a frame for the about information
        about_frame = ttk.Frame(self.about_tab, padding="20")
        about_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        
        # App title and version
        title_label = ttk.Label(about_frame, text="SylvaFine", font=("Helvetica", 16, "bold"))
        title_label.grid(row=0, column=0, pady=5)
        
        version_label = ttk.Label(about_frame, text="Version 1.0.0")
        version_label.grid(row=1, column=0, pady=5)
        
        # Creator information
        creator_label = ttk.Label(about_frame, text="Created by Morgan Roberts MSW")
        creator_label.grid(row=2, column=0, pady=5)
        
        # Description
        description = "SylvaFine is an advanced language model fine-tuning tool that allows you to customize and train AI models for your specific needs. With an intuitive interface and powerful features, SylvaFine makes it easy to fine-tune language models using your own datasets."
        desc_label = ttk.Label(about_frame, text=description, wraplength=500, justify="center")
        desc_label.grid(row=3, column=0, pady=10)
        
        # Links
        link_frame = ttk.Frame(about_frame)
        link_frame.grid(row=4, column=0, pady=10)
        
        ttk.Button(link_frame, text="Documentation", command=lambda: webbrowser.open("https://platform.openai.com/docs/guides/fine-tuning")).grid(row=0, column=0, padx=5)
        ttk.Button(link_frame, text="OpenAI Website", command=lambda: webbrowser.open("https://openai.com")).grid(row=0, column=1, padx=5)
        
        # Configure grid weights
        self.about_tab.columnconfigure(0, weight=1)
        self.about_tab.rowconfigure(0, weight=1)
        
    def log_message(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.see(tk.END)
        
        # Also update history if it's a response
        if "Generated response:" in message:
            self.add_to_history(self.test_prompt.get(), message.replace("Generated response: ", ""))
        
    def browse_dataset(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("JSONL files", "*.jsonl"), ("All files", "*.*")]
        )
        if file_path:
            self.dataset_path.set(file_path)
            self.log_message(f"Dataset selected: {file_path}")
            
    def validate_inputs(self):
        try:
            epochs = int(self.epochs.get())
            batch_size = int(self.batch_size.get())
            learning_rate = float(self.learning_rate.get())
            if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
                raise ValueError
            return True
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numbers for epochs, batch size, and learning rate")
            return False
            
    def start_training(self):
        if not self.validate_inputs():
            return
            
        if not self.dataset_path.get():
            messagebox.showerror("Error", "Please select a dataset file first")
            return
            
        if not os.path.exists(self.dataset_path.get()):
            messagebox.showerror("Error", "Dataset file does not exist")
            return
            
        if self.training_active:
            messagebox.showwarning("Warning", "Training is already in progress")
            return
            
        self.training_active = True
        self.log_message("Starting training process...")
        
        # Reset training metrics
        self.training_metrics = {"loss": [], "accuracy": []}
        
        # Create training thread
        training_thread = threading.Thread(target=self.run_training)
        training_thread.daemon = True
        training_thread.start()
        
    def run_training(self):
        try:
            # Simulate training process (in a real application, this would interact with OpenAI API)
            self.log_message("Loading dataset...")
            time.sleep(2)  # Simulate loading
            
            epochs = int(self.epochs.get())
            batch_size = int(self.batch_size.get())
            
            for epoch in range(epochs):
                self.log_message(f"Starting epoch {epoch + 1}/{epochs}")
                for batch in range(10):  # Simulate 10 batches per epoch
                    self.log_message(f"Processing batch {batch + 1}/10")
                    
                    # Simulate metrics for visualization
                    loss = 1.0 - (epoch * 0.2 + batch * 0.02)
                    if loss < 0.2: loss = 0.2
                    accuracy = 0.5 + (epoch * 0.1 + batch * 0.01)
                    if accuracy > 0.95: accuracy = 0.95
                    
                    self.training_metrics["loss"].append(loss)
                    self.training_metrics["accuracy"].append(accuracy)
                    
                    # Update plot if we're on the visualization tab
                    if self.notebook.index("current") == 1:  # Visualization tab
                        self.update_plot()
                    
                    time.sleep(1)  # Simulate processing
                
            self.log_message("Training completed successfully!")
            self.log_message("Model ready for testing")
            
            # Final plot update
            self.update_plot()
            
        except Exception as e:
            self.log_message(f"Error during training: {str(e)}")
        finally:
            self.training_active = False
    
    def stop_training(self):
        if not self.training_active:
            messagebox.showinfo("Info", "No training is currently active")
            return
            
        self.log_message("Stopping training...")
        self.training_active = False
        
    def validate_api_key(self):
        # Add your OpenAI API key validation logic here
        # For demonstration purposes, this method always returns True
        return True
            
    def generate_response(self):
        if not self.test_prompt.get():
            messagebox.showerror("Error", "Please enter a test prompt")
            return
            
        try:
            if not self.validate_api_key():
                return
                
            self.log_message("Generating response...")
            
            # Use OpenAI's API to generate a response
            try:
                # Create a chat completion using the client
                response = client.chat.completions.create(
                    model=self.selected_model.get(),
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant trained on emotional support prompts and responses."},
                        {"role": "user", "content": self.test_prompt.get()}
                    ],
                    temperature=0.7,
                    max_tokens=150
                )
                
                # Extract the response content
                response_text = response.choices[0].message.content.strip()
                self.log_message(f"Generated response: {response_text}")
                
            except Exception as e:
                self.log_message(f"Error generating response: {str(e)}")
                
        except Exception as e:
            self.log_message(f"Error: {str(e)}")
    
    def save_response(self):
        if not hasattr(self, 'last_response') or not self.last_response:
            messagebox.showinfo("Info", "No response to save")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(f"Prompt: {self.test_prompt.get()}\n\n")
                    f.write(f"Response: {self.last_response}\n")
                self.log_message(f"Response saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save response: {str(e)}")
    
    def add_to_history(self, prompt, response):
        # Add to response history list
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.response_history.append({
            "timestamp": timestamp,
            "prompt": prompt,
            "response": response,
            "model": self.selected_model.get()
        })
        
        # Update history display if we're on the history tab
        self.update_history_display()
        
        # Save as last response for single-response saving
        self.last_response = response
    
    def update_history_display(self):
        # Clear current history display
        self.history_text.config(state=tk.NORMAL)
        self.history_text.delete(1.0, tk.END)
        
        # Add each response to the display
        for item in self.response_history:
            self.history_text.insert(tk.END, f"Time: {item['timestamp']}\n")
            self.history_text.insert(tk.END, f"Model: {item['model']}\n")
            self.history_text.insert(tk.END, f"Prompt: {item['prompt']}\n")
            self.history_text.insert(tk.END, f"Response: {item['response']}\n")
            self.history_text.insert(tk.END, "\n" + "-"*50 + "\n\n")
        
        self.history_text.config(state=tk.DISABLED)
    
    def clear_history(self):
        if messagebox.askyesno("Confirm", "Are you sure you want to clear the response history?"):
            self.response_history = []
            self.update_history_display()
            self.log_message("Response history cleared")
    
    def export_responses(self):
        if not self.response_history:
            messagebox.showinfo("Info", "No responses to export")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Timestamp", "Model", "Prompt", "Response"])
                    for item in self.response_history:
                        writer.writerow([item['timestamp'], item['model'], item['prompt'], item['response']])
                self.log_message(f"Responses exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export responses: {str(e)}")
    
    def update_plot(self):
        """Update all visualization plots with current data"""
        # Update loss plot
        self.update_loss_plot()
        
        # Update accuracy plot
        self.update_accuracy_plot()
        
        # Update learning rate plot
        self.update_lr_plot()
        
        # Update validation plot
        self.update_validation_plot()
    
    def update_loss_plot(self):
        """Update the loss plot"""
        # Clear the current plot
        self.loss_figure.clear()
        ax = self.loss_figure.add_subplot(111)
        
        # Set up the plot
        ax.set_title('Training Loss')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Loss')
        
        # Plot the data if available
        if self.training_metrics['loss']:
            batches = list(range(len(self.training_metrics['loss'])))
            ax.plot(batches, self.training_metrics['loss'], 'b-', label='Training Loss')
            ax.legend()
            
        # Redraw the canvas
        self.loss_canvas.draw()
    
    def update_accuracy_plot(self):
        """Update the accuracy plot"""
        # Clear the current plot
        self.accuracy_figure.clear()
        ax = self.accuracy_figure.add_subplot(111)
        
        # Set up the plot
        ax.set_title('Training Accuracy')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Accuracy')
        
        # Plot the data if available
        if self.training_metrics['accuracy']:
            batches = list(range(len(self.training_metrics['accuracy'])))
            ax.plot(batches, self.training_metrics['accuracy'], 'g-', label='Accuracy')
            ax.legend()
            
        # Redraw the canvas
        self.accuracy_canvas.draw()
    
    def update_lr_plot(self):
        """Update the learning rate plot"""
        # Clear the current plot
        self.lr_figure.clear()
        ax = self.lr_figure.add_subplot(111)
        
        # Set up the plot
        ax.set_title('Learning Rate')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Learning Rate')
        
        # Plot the data if available
        if self.training_metrics['learning_rate']:
            batches = list(range(len(self.training_metrics['learning_rate'])))
            ax.plot(batches, self.training_metrics['learning_rate'], 'r-', label='Learning Rate')
            ax.legend()
            
        # Redraw the canvas
        self.lr_canvas.draw()
    
    def update_validation_plot(self):
        """Update the validation plot"""
        # Clear the current plot
        self.validation_figure.clear()
        ax = self.validation_figure.add_subplot(111)
        
        # Set up the plot
        ax.set_title('Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        
        # Plot the data if available
        if self.training_metrics['validation_loss']:
            epochs = list(range(len(self.training_metrics['validation_loss'])))
            ax.plot(epochs, self.training_metrics['validation_loss'], 'm-', label='Validation Loss')
            ax.legend()
            
        # Redraw the canvas
        self.validation_canvas.draw()
    
    def clear_plots(self):
        """Clear all plots"""
        # Clear all metrics data
        for key in self.training_metrics:
            self.training_metrics[key] = []
            
        # Update all plots
        self.update_plot()
        self.log_message("Cleared all plots")
    
    def export_plot(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Determine which figure to export based on current tab
                current_tab = self.notebook.index("current")
                if current_tab == 3:  # Visualization tab
                    # Get the current visualization tab
                    viz_notebook = self.viz_tab.winfo_children()[0].winfo_children()[0]
                    viz_tab_id = viz_notebook.index("current")
                    
                    if viz_tab_id == 0:  # Loss tab
                        self.loss_figure.savefig(file_path)
                    elif viz_tab_id == 1:  # Accuracy tab
                        self.accuracy_figure.savefig(file_path)
                    elif viz_tab_id == 2:  # Learning Rate tab
                        self.lr_figure.savefig(file_path)
                    elif viz_tab_id == 3:  # Validation tab
                        self.validation_figure.savefig(file_path)
                elif current_tab == 4:  # Evaluation tab
                    # Get the current evaluation tab
                    eval_notebook = self.evaluation_tab.winfo_children()[0].winfo_children()[3]
                    eval_tab_id = eval_notebook.index("current")
                    
                    if eval_tab_id == 1:  # Metrics tab
                        self.metrics_figure.savefig(file_path)
                    elif eval_tab_id == 2:  # Confusion Matrix tab
                        self.confusion_figure.savefig(file_path)
                
                self.log_message(f"Plot exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export plot: {str(e)}")
    
    def browse_test_dataset(self):
        """Browse for a test dataset file"""
        file_path = filedialog.askopenfilename(
            filetypes=[("JSONL files", "*.jsonl"), ("All files", "*.*")],
            title="Select Test Dataset File"
        )
        
        if file_path:
            self.test_dataset_path.set(file_path)
            self.log_message(f"Selected test dataset: {file_path}")
    
    def run_evaluation(self):
        """Run evaluation on a test dataset"""
        if not self.test_dataset_path.get() or not os.path.exists(self.test_dataset_path.get()):
            messagebox.showerror("Error", "Please select a valid test dataset file")
            return
            
        if not self.validate_api_key():
            return
            
        # Confirm with user
        if not messagebox.askyesno("Confirm Evaluation", 
                                 "This will run evaluation on the test dataset and may incur API costs. Continue?"):
            return
            
        try:
            # Load test dataset
            test_examples = []
            with jsonlines.open(self.test_dataset_path.get(), 'r') as reader:
                for obj in reader:
                    test_examples.append(obj)
                    
            if not test_examples:
                messagebox.showerror("Error", "Test dataset is empty")
                return
                
            self.log_message(f"Loaded {len(test_examples)} examples from test dataset")
            
            # Clear previous results
            self.test_results = []
            self.evaluation_metrics = {
                "bleu": [],
                "meteor": [],
                "edit_distance": [],
                "response_length": [],
                "response_time": []
            }
            
            # Create progress dialog
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Evaluation Progress")
            progress_window.geometry("400x150")
            progress_window.transient(self.root)
            progress_window.grab_set()
            
            progress_label = ttk.Label(progress_window, text="Evaluating examples...")
            progress_label.pack(pady=10)
            
            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=len(test_examples))
            progress_bar.pack(fill=tk.X, padx=20, pady=10)
            
            status_label = ttk.Label(progress_window, text="Starting...")
            status_label.pack(pady=10)
            
            # Start evaluation in a separate thread
            def run_eval_thread():
                for i, example in enumerate(test_examples):
                    # Update progress
                    progress_var.set(i + 1)
                    status_label.config(text=f"Processing example {i + 1} of {len(test_examples)}")
                    progress_window.update()
                    
                    # Extract messages
                    if "messages" in example:
                        messages = example["messages"]
                        system_msg = ""
                        user_msg = ""
                        expected_response = ""
                        
                        for msg in messages:
                            if msg["role"] == "system":
                                system_msg = msg["content"]
                            elif msg["role"] == "user":
                                user_msg = msg["content"]
                            elif msg["role"] == "assistant":
                                expected_response = msg["content"]
                        
                        if user_msg and expected_response:
                            # Generate response using the model
                            start_time = time.time()
                            
                            api_messages = []
                            if system_msg:
                                api_messages.append({"role": "system", "content": system_msg})
                            api_messages.append({"role": "user", "content": user_msg})
                            
                            try:
                                response = client.chat.completions.create(
                                    model=self.selected_model.get(),
                                    messages=api_messages,
                                    temperature=0.7,
                                    max_tokens=1024
                                )
                                
                                actual_response = response.choices[0].message.content
                                response_time = time.time() - start_time
                                
                                # Calculate metrics
                                bleu_score = self.calculate_bleu(expected_response, actual_response)
                                meteor_score = self.calculate_meteor(expected_response, actual_response)
                                edit_dist = self.calculate_edit_distance(expected_response, actual_response)
                                
                                # Store results
                                result = {
                                    "user_prompt": user_msg,
                                    "expected_response": expected_response,
                                    "actual_response": actual_response,
                                    "bleu": bleu_score,
                                    "meteor": meteor_score,
                                    "edit_distance": edit_dist,
                                    "response_time": response_time,
                                    "response_length": len(actual_response)
                                }
                                
                                self.test_results.append(result)
                                
                                # Update metrics
                                self.evaluation_metrics["bleu"].append(bleu_score)
                                self.evaluation_metrics["meteor"].append(meteor_score)
                                self.evaluation_metrics["edit_distance"].append(edit_dist)
                                self.evaluation_metrics["response_time"].append(response_time)
                                self.evaluation_metrics["response_length"].append(len(actual_response))
                                
                            except Exception as e:
                                self.log_message(f"Error generating response for example {i+1}: {str(e)}")
                
                # Close progress window
                progress_window.destroy()
                
                # Update results display
                self.update_evaluation_results()
                self.log_message(f"Evaluation completed on {len(self.test_results)} examples")
                
            # Start the thread
            threading.Thread(target=run_eval_thread, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run evaluation: {str(e)}")
            self.log_message(f"Error running evaluation: {str(e)}")
    
    def run_single_test(self):
        """Run a single test with the current prompt"""
        if not self.test_prompt.get().strip():
            messagebox.showerror("Error", "Please enter a test prompt")
            return
            
        if not self.validate_api_key():
            return
            
        try:
            # Get the prompt
            prompt = self.test_prompt.get().strip()
            
            # Generate response
            self.log_message("Generating response for single test...")
            
            # Create messages for API call
            messages = [{"role": "user", "content": prompt}]
            
            # Call OpenAI API
            start_time = time.time()
            response = client.chat.completions.create(
                model=self.selected_model.get(),
                messages=messages,
                temperature=0.7,
                max_tokens=1024
            )
            
            # Extract response
            actual_response = response.choices[0].message.content
            response_time = time.time() - start_time
            
            # Create result dialog
            result_window = tk.Toplevel(self.root)
            result_window.title("Test Result")
            result_window.geometry("600x500")
            
            # Create frame for result
            frame = ttk.Frame(result_window, padding="10")
            frame.pack(fill=tk.BOTH, expand=True)
            
            # Show prompt and response
            ttk.Label(frame, text="Prompt:").grid(row=0, column=0, sticky=tk.W, pady=5)
            prompt_text = scrolledtext.ScrolledText(frame, height=5, width=60, wrap=tk.WORD)
            prompt_text.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
            prompt_text.insert(tk.END, prompt)
            prompt_text.config(state=tk.DISABLED)
            
            ttk.Label(frame, text="Response:").grid(row=2, column=0, sticky=tk.W, pady=5)
            response_text = scrolledtext.ScrolledText(frame, height=10, width=60, wrap=tk.WORD)
            response_text.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)
            response_text.insert(tk.END, actual_response)
            response_text.config(state=tk.DISABLED)
            
            # Show metrics
            metrics_text = f"Response Time: {response_time:.2f} seconds\nResponse Length: {len(actual_response)} characters"
            ttk.Label(frame, text="Metrics:").grid(row=4, column=0, sticky=tk.W, pady=5)
            metrics_label = ttk.Label(frame, text=metrics_text)
            metrics_label.grid(row=5, column=0, sticky=tk.W, pady=5)
            
            # Add buttons
            button_frame = ttk.Frame(frame)
            button_frame.grid(row=6, column=0, sticky=(tk.W, tk.E), pady=10)
            
            ttk.Button(button_frame, text="Save to History", 
                      command=lambda: self.add_to_history(prompt, actual_response) or result_window.destroy()).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Close", command=result_window.destroy).pack(side=tk.LEFT, padx=5)
            
            # Configure grid weights
            frame.columnconfigure(0, weight=1)
            
            self.log_message("Single test completed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run test: {str(e)}")
            self.log_message(f"Error running test: {str(e)}")
    
    def update_evaluation_results(self):
        """Update the evaluation results display"""
        if not self.test_results:
            return
            
        # Update results text
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        # Create a table header
        header = "Example\tBLEU\tMETEOR\tEdit Distance\tResponse Time\tResponse Length\n"
        header += "-" * 80 + "\n"
        self.results_text.insert(tk.END, header)
        
        # Add each result
        for i, result in enumerate(self.test_results):
            row = f"{i+1}\t{result['bleu']:.4f}\t{result['meteor']:.4f}\t{result['edit_distance']}\t"
            row += f"{result['response_time']:.2f}s\t{result['response_length']}\n"
            self.results_text.insert(tk.END, row)
            
        # Add summary
        summary = "\nSummary:\n"
        summary += "-" * 80 + "\n"
        
        # Calculate averages
        avg_bleu = sum(self.evaluation_metrics["bleu"]) / len(self.evaluation_metrics["bleu"]) if self.evaluation_metrics["bleu"] else 0
        avg_meteor = sum(self.evaluation_metrics["meteor"]) / len(self.evaluation_metrics["meteor"]) if self.evaluation_metrics["meteor"] else 0
        avg_edit = sum(self.evaluation_metrics["edit_distance"]) / len(self.evaluation_metrics["edit_distance"]) if self.evaluation_metrics["edit_distance"] else 0
        avg_time = sum(self.evaluation_metrics["response_time"]) / len(self.evaluation_metrics["response_time"]) if self.evaluation_metrics["response_time"] else 0
        avg_length = sum(self.evaluation_metrics["response_length"]) / len(self.evaluation_metrics["response_length"]) if self.evaluation_metrics["response_length"] else 0
        
        summary += f"Average BLEU: {avg_bleu:.4f}\n"
        summary += f"Average METEOR: {avg_meteor:.4f}\n"
        summary += f"Average Edit Distance: {avg_edit:.2f}\n"
        summary += f"Average Response Time: {avg_time:.2f}s\n"
        summary += f"Average Response Length: {avg_length:.2f} characters\n"
        
        self.results_text.insert(tk.END, summary)
        self.results_text.config(state=tk.DISABLED)
        
        # Update metrics plot
        self.update_metrics_plot()
        
        # Update confusion matrix
        self.update_confusion_matrix()
    
    def update_metrics_plot(self):
        """Update the metrics plot"""
        if not self.test_results:
            return
            
        # Clear the current plot
        self.metrics_figure.clear()
        
        # Create subplots
        gs = GridSpec(2, 2, figure=self.metrics_figure)
        ax1 = self.metrics_figure.add_subplot(gs[0, 0])  # BLEU
        ax2 = self.metrics_figure.add_subplot(gs[0, 1])  # METEOR
        ax3 = self.metrics_figure.add_subplot(gs[1, 0])  # Edit Distance
        ax4 = self.metrics_figure.add_subplot(gs[1, 1])  # Response Time
        
        # Plot BLEU scores
        ax1.set_title('BLEU Scores')
        ax1.boxplot(self.evaluation_metrics["bleu"])
        ax1.set_ylim(0, 1)
        
        # Plot METEOR scores
        ax2.set_title('METEOR Scores')
        ax2.boxplot(self.evaluation_metrics["meteor"])
        ax2.set_ylim(0, 1)
        
        # Plot Edit Distances
        ax3.set_title('Edit Distances')
        ax3.boxplot(self.evaluation_metrics["edit_distance"])
        
        # Plot Response Times
        ax4.set_title('Response Times (s)')
        ax4.boxplot(self.evaluation_metrics["response_time"])
        
        # Adjust layout
        self.metrics_figure.tight_layout()
        
        # Redraw the canvas
        self.metrics_canvas.draw()
    
    def update_confusion_matrix(self):
        """Update the confusion matrix visualization"""
        if not self.test_results or not self.confusion_matrix_data:
            # No confusion matrix data yet
            return
            
        # Clear the current plot
        self.confusion_figure.clear()
        ax = self.confusion_figure.add_subplot(111)
        
        # Plot the confusion matrix
        cax = ax.matshow(self.confusion_matrix_data, cmap='Blues')
        self.confusion_figure.colorbar(cax)
        
        # Set labels
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        # Set ticks
        if self.evaluation_categories:
            ax.set_xticks(range(len(self.evaluation_categories)))
            ax.set_yticks(range(len(self.evaluation_categories)))
            ax.set_xticklabels(self.evaluation_categories, rotation=45, ha='right')
            ax.set_yticklabels(self.evaluation_categories)
            
        # Redraw the canvas
        self.confusion_canvas.draw()
    
    def export_evaluation_results(self):
        """Export evaluation results to a CSV file"""
        if not self.test_results:
            messagebox.showerror("Error", "No evaluation results to export")
            return
            
        # Ask for file location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export Evaluation Results"
        )
        
        if not file_path:
            return  # User cancelled
            
        try:
            # Write results to CSV
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                writer.writerow(["Example", "User Prompt", "Expected Response", "Actual Response", 
                               "BLEU", "METEOR", "Edit Distance", "Response Time", "Response Length"])
                
                # Write each result
                for i, result in enumerate(self.test_results):
                    writer.writerow([
                        i+1,
                        result["user_prompt"],
                        result["expected_response"],
                        result["actual_response"],
                        result["bleu"],
                        result["meteor"],
                        result["edit_distance"],
                        result["response_time"],
                        result["response_length"]
                    ])
                    
            self.log_message(f"Exported evaluation results to {file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export results: {str(e)}")
            self.log_message(f"Error exporting results: {str(e)}")
    
    def clear_evaluation_results(self):
        """Clear all evaluation results"""
        if not self.test_results:
            return
            
        # Confirm with user
        if not messagebox.askyesno("Confirm Clear", "Are you sure you want to clear all evaluation results?"):
            return
            
        # Clear results
        self.test_results = []
        self.evaluation_metrics = {
            "bleu": [],
            "meteor": [],
            "edit_distance": [],
            "response_length": [],
            "response_time": []
        }
        
        # Clear display
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.DISABLED)
        
        # Clear plots
        self.metrics_figure.clear()
        self.metrics_canvas.draw()
        
        self.confusion_figure.clear()
        self.confusion_canvas.draw()
        
        self.log_message("Cleared all evaluation results")
    
    def calculate_bleu(self, reference, hypothesis):
        """Calculate BLEU score between reference and hypothesis"""
        reference_tokens = nltk.word_tokenize(reference.lower())
        hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())
        
        # Use smoothing to avoid zero scores when there are no n-gram matches
        smoothing = SmoothingFunction().method1
        
        # Calculate BLEU score
        try:
            return sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=smoothing)
        except Exception:
            return 0.0
    
    def calculate_meteor(self, reference, hypothesis):
        """Calculate METEOR score between reference and hypothesis"""
        reference_tokens = nltk.word_tokenize(reference.lower())
        hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())
        
        # Calculate METEOR score
        try:
            return meteor_score([reference_tokens], hypothesis_tokens)
        except Exception:
            return 0.0
    
    def calculate_edit_distance(self, reference, hypothesis):
        """Calculate Levenshtein edit distance between reference and hypothesis"""
        try:
            return edit_distance(reference.lower(), hypothesis.lower())
        except Exception:
            return 0
    
    def change_theme(self, event=None):
        theme = self.theme_var.get()
        
        if theme == "Dark":
            # Apply dark theme
            self.style.configure("TFrame", background="#333333")
            self.style.configure("TLabel", background="#333333", foreground="#FFFFFF")
            self.style.configure("TButton", background="#555555", foreground="#FFFFFF")
            self.style.configure("TLabelframe", background="#333333", foreground="#FFFFFF")
            self.style.configure("TLabelframe.Label", background="#333333", foreground="#FFFFFF")
            
            # Configure text widgets
            self.status_text.config(bg="#222222", fg="#FFFFFF")
            if hasattr(self, 'history_text'):
                self.history_text.config(bg="#222222", fg="#FFFFFF")
                
        elif theme == "Light":
            # Apply light theme
            self.style.configure("TFrame", background="#F0F0F0")
            self.style.configure("TLabel", background="#F0F0F0", foreground="#000000")
            self.style.configure("TButton", background="#E0E0E0", foreground="#000000")
            self.style.configure("TLabelframe", background="#F0F0F0", foreground="#000000")
            self.style.configure("TLabelframe.Label", background="#F0F0F0", foreground="#000000")
            
            # Configure text widgets
            self.status_text.config(bg="#FFFFFF", fg="#000000")
            if hasattr(self, 'history_text'):
                self.history_text.config(bg="#FFFFFF", fg="#000000")
                
        else:  # Default
            # Reset to default theme
            self.style = ttk.Style()
            
            # Configure text widgets
            self.status_text.config(bg="white", fg="black")
            if hasattr(self, 'history_text'):
                self.history_text.config(bg="white", fg="black")
        
        self.log_message(f"Theme changed to {theme}")
    
    def update_api_key(self):
        # Create a dialog to update the API key
        dialog = tk.Toplevel(self.root)
        dialog.title("Update API Key")
        dialog.geometry("400x150")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Enter your OpenAI API Key:").pack(pady=10)
        
        api_key_var = tk.StringVar()
        if os.getenv('OPENAI_API_KEY'):
            api_key_var.set(os.getenv('OPENAI_API_KEY'))
            
        ttk.Entry(dialog, textvariable=api_key_var, width=40, show="*").pack(pady=5)
        
        def save_key():
            new_key = api_key_var.get().strip()
            if not new_key:
                messagebox.showerror("Error", "API Key cannot be empty")
                return
                
            # Update the environment variable
            os.environ['OPENAI_API_KEY'] = new_key
            
            # Update the client
            global client
            client = openai.OpenAI(
                api_key=new_key,
                base_url="https://api.openai.com/v1"
            )
            
            self.log_message("API Key updated successfully")
            dialog.destroy()
            
        ttk.Button(dialog, text="Save", command=save_key).pack(pady=10)
    
    def save_settings(self):
        # Save current settings to a config file
        config = {
            "theme": self.theme_var.get(),
            "model": self.selected_model.get(),
            "epochs": self.epochs.get(),
            "batch_size": self.batch_size.get(),
            "learning_rate": self.learning_rate.get()
        }
        
        try:
            with open("sylvafine_config.json", "w") as f:
                json.dump(config, f, indent=4)
            self.log_message("Settings saved successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {str(e)}")
    
    def load_settings(self):
        # Load settings from config file if it exists
        if os.path.exists("sylvafine_config.json"):
            try:
                with open("sylvafine_config.json", "r") as f:
                    config = json.load(f)
                    
                # Apply settings
                if "theme" in config:
                    self.theme_var.set(config["theme"])
                    self.change_theme()
                    
                if "model" in config:
                    self.selected_model.set(config["model"])
                    
                if "epochs" in config:
                    self.epochs.set(config["epochs"])
                    
                if "batch_size" in config:
                    self.batch_size.set(config["batch_size"])
                    
                if "learning_rate" in config:
                    self.learning_rate.set(config["learning_rate"])
                    
                self.log_message("Settings loaded successfully")
            except Exception as e:
                self.log_message(f"Error loading settings: {str(e)}")
    
    def new_project(self):
        if messagebox.askyesno("Confirm", "Start a new project? This will clear current data."):
            self.dataset_path.set("")
            self.test_prompt.set("")
            self.response_history = []
            self.training_metrics = {"loss": [], "accuracy": []}
            self.update_history_display()
            self.update_plot()
            self.log_message("New project started")
    
    def open_project(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, "r") as f:
                    project_data = json.load(f)
                    
                # Load project data
                if "dataset_path" in project_data and os.path.exists(project_data["dataset_path"]):
                    self.dataset_path.set(project_data["dataset_path"])
                    
                if "responses" in project_data:
                    self.response_history = project_data["responses"]
                    self.update_history_display()
                    
                if "training_metrics" in project_data:
                    self.training_metrics = project_data["training_metrics"]
                    self.update_plot()
                    
                self.log_message(f"Project loaded from {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load project: {str(e)}")
    
    def save_project(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                project_data = {
                    "dataset_path": self.dataset_path.get(),
                    "responses": self.response_history,
                    "training_metrics": self.training_metrics,
                    "settings": {
                        "model": self.selected_model.get(),
                        "epochs": self.epochs.get(),
                        "batch_size": self.batch_size.get(),
                        "learning_rate": self.learning_rate.get()
                    }
                }
                
                with open(file_path, "w") as f:
                    json.dump(project_data, f, indent=4)
                    
                self.log_message(f"Project saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save project: {str(e)}")
    
    def validate_dataset(self):
        if not self.dataset_path.get() or not os.path.exists(self.dataset_path.get()):
            messagebox.showerror("Error", "Please select a valid dataset file first")
            return
            
        try:
            valid_count = 0
            invalid_count = 0
            invalid_lines = []
            error_details = []
            
            # First try to read the file line by line with more robust error handling
            with open(self.dataset_path.get(), 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
                for i, line in enumerate(lines):
                    try:
                        line = line.strip()
                        if not line:  # Skip empty lines
                            continue
                            
                        # Try to parse the JSON
                        data = json.loads(line)
                        
                        # Check if it has the expected format for fine-tuning
                        if "messages" in data and isinstance(data["messages"], list):
                            valid_count += 1
                        else:
                            invalid_count += 1
                            invalid_lines.append(i + 1)
                            error_details.append(f"Line {i+1}: Missing 'messages' field or not a list")
                            
                    except json.JSONDecodeError as json_err:
                        invalid_count += 1
                        invalid_lines.append(i + 1)
                        error_details.append(f"Line {i+1}: JSON parsing error - {str(json_err)}")
                    except Exception as line_err:
                        invalid_count += 1
                        invalid_lines.append(i + 1)
                        error_details.append(f"Line {i+1}: Unexpected error - {str(line_err)}")
            
            # Show results
            if invalid_count == 0:
                messagebox.showinfo("Validation Result", 
                                   f"Dataset is valid! Contains {valid_count} valid examples ready for fine-tuning.")
                self.log_message(f"Dataset validation successful: {valid_count} valid examples")
            else:
                # Create a detailed error report
                error_report = f"Dataset contains {invalid_count} invalid examples out of {valid_count + invalid_count} total.\n\n"
                error_report += "Detailed errors:\n"
                for i, error in enumerate(error_details[:10]):
                    error_report += f"{error}\n"
                    
                if len(error_details) > 10:
                    error_report += f"...and {len(error_details) - 10} more errors.\n\n"
                    
                error_report += "\nCommon fixes:\n"
                error_report += "1. Ensure each line is a valid JSON object\n"
                error_report += "2. All property names must be in double quotes\n"
                error_report += "3. Each line must have a 'messages' array\n"
                error_report += "4. The 'messages' array should contain objects with 'role' and 'content'\n"
                
                # Show the error report in a scrollable dialog
                self.show_error_report("Dataset Validation Errors", error_report)
                self.log_message(f"Dataset validation found {invalid_count} errors")
                
        except Exception as e:
            detailed_error = f"Error validating dataset: {str(e)}\n\n"
            detailed_error += "Possible causes:\n"
            detailed_error += "1. File encoding issues - ensure the file is UTF-8 encoded\n"
            detailed_error += "2. File format issues - ensure it's a proper JSONL file\n"
            detailed_error += "3. File access issues - ensure you have read permissions\n"
            
            messagebox.showerror("Error", detailed_error)
            self.log_message(f"Dataset validation error: {str(e)}")
    
    def show_error_report(self, title, error_text):
        """Show a detailed error report in a scrollable dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.geometry("600x400")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Add a scrollable text area
        text_area = scrolledtext.ScrolledText(dialog, wrap=tk.WORD, width=70, height=20)
        text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_area.insert(tk.END, error_text)
        text_area.config(state=tk.DISABLED)
        
        # Add a close button
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)
    
    def view_dataset_stats(self):
        if not self.dataset_path.get() or not os.path.exists(self.dataset_path.get()):
            messagebox.showerror("Error", "Please select a valid dataset file first")
            return
            
        try:
            self.log_message(f"Analyzing dataset: {self.dataset_path.get()}")
            
            # Initialize statistics variables
            total_examples = 0
            valid_examples = 0
            invalid_examples = 0
            total_tokens = 0
            prompt_tokens = 0
            completion_tokens = 0
            role_counts = {"system": 0, "user": 0, "assistant": 0, "other": 0}
            avg_messages_per_example = 0
            message_counts = []
            
            # Process the file line by line with robust error handling
            with open(self.dataset_path.get(), 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
                for i, line in enumerate(lines):
                    try:
                        line = line.strip()
                        if not line:  # Skip empty lines
                            continue
                            
                        total_examples += 1
                        
                        # Try to parse the JSON
                        data = json.loads(line)
                        
                        # Check if it has the expected format for fine-tuning
                        if "messages" in data and isinstance(data["messages"], list):
                            valid_examples += 1
                            messages = data["messages"]
                            message_counts.append(len(messages))
                            
                            for msg in messages:
                                role = msg.get("role", "other")
                                content = msg.get("content", "")
                                
                                # Count tokens (simple word-based estimate)
                                words = len(content.split())
                                total_tokens += words
                                
                                # Track by role
                                if role in role_counts:
                                    role_counts[role] += 1
                                else:
                                    role_counts["other"] += 1
                                    
                                # Track tokens by role
                                if role == "user":
                                    prompt_tokens += words
                                elif role == "assistant":
                                    completion_tokens += words
                        else:
                            invalid_examples += 1
                            
                    except json.JSONDecodeError:
                        invalid_examples += 1
                    except Exception:
                        invalid_examples += 1
            
            # Calculate averages
            if message_counts:
                avg_messages_per_example = sum(message_counts) / len(message_counts)
            
            # Create a stats dialog with tabs for different statistics
            dialog = tk.Toplevel(self.root)
            dialog.title("Dataset Statistics")
            dialog.geometry("500x400")
            dialog.transient(self.root)
            dialog.grab_set()
            
            # Create notebook for tabs
            notebook = ttk.Notebook(dialog)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Summary tab
            summary_tab = ttk.Frame(notebook, padding=10)
            notebook.add(summary_tab, text="Summary")
            
            ttk.Label(summary_tab, text="Dataset Summary", font=("Helvetica", 14, "bold")).grid(row=0, column=0, columnspan=2, pady=10)
            
            ttk.Label(summary_tab, text=f"File:").grid(row=1, column=0, sticky=tk.W, pady=5)
            ttk.Label(summary_tab, text=f"{os.path.basename(self.dataset_path.get())}").grid(row=1, column=1, sticky=tk.W, pady=5)
            
            ttk.Label(summary_tab, text=f"Total examples:").grid(row=2, column=0, sticky=tk.W, pady=5)
            ttk.Label(summary_tab, text=f"{total_examples}").grid(row=2, column=1, sticky=tk.W, pady=5)
            
            ttk.Label(summary_tab, text=f"Valid examples:").grid(row=3, column=0, sticky=tk.W, pady=5)
            ttk.Label(summary_tab, text=f"{valid_examples}").grid(row=3, column=1, sticky=tk.W, pady=5)
            
            ttk.Label(summary_tab, text=f"Invalid examples:").grid(row=4, column=0, sticky=tk.W, pady=5)
            ttk.Label(summary_tab, text=f"{invalid_examples}").grid(row=4, column=1, sticky=tk.W, pady=5)
            
            ttk.Label(summary_tab, text=f"Avg messages per example:").grid(row=5, column=0, sticky=tk.W, pady=5)
            ttk.Label(summary_tab, text=f"{avg_messages_per_example:.2f}").grid(row=5, column=1, sticky=tk.W, pady=5)
            
            # Token statistics tab
            token_tab = ttk.Frame(notebook, padding=10)
            notebook.add(token_tab, text="Token Stats")
            
            ttk.Label(token_tab, text="Token Statistics", font=("Helvetica", 14, "bold")).grid(row=0, column=0, columnspan=2, pady=10)
            
            ttk.Label(token_tab, text=f"Estimated total tokens:").grid(row=1, column=0, sticky=tk.W, pady=5)
            ttk.Label(token_tab, text=f"{total_tokens:,}").grid(row=1, column=1, sticky=tk.W, pady=5)
            
            ttk.Label(token_tab, text=f"Estimated prompt tokens:").grid(row=2, column=0, sticky=tk.W, pady=5)
            ttk.Label(token_tab, text=f"{prompt_tokens:,}").grid(row=2, column=1, sticky=tk.W, pady=5)
            
            ttk.Label(token_tab, text=f"Estimated completion tokens:").grid(row=3, column=0, sticky=tk.W, pady=5)
            ttk.Label(token_tab, text=f"{completion_tokens:,}").grid(row=3, column=1, sticky=tk.W, pady=5)
            
            ttk.Label(token_tab, text=f"Prompt/completion ratio:").grid(row=4, column=0, sticky=tk.W, pady=5)
            ratio = prompt_tokens / completion_tokens if completion_tokens > 0 else 0
            ttk.Label(token_tab, text=f"{ratio:.2f}").grid(row=4, column=1, sticky=tk.W, pady=5)
            
            # Role statistics tab
            role_tab = ttk.Frame(notebook, padding=10)
            notebook.add(role_tab, text="Role Stats")
            
            ttk.Label(role_tab, text="Message Role Statistics", font=("Helvetica", 14, "bold")).grid(row=0, column=0, columnspan=2, pady=10)
            
            row = 1
            for role, count in role_counts.items():
                ttk.Label(role_tab, text=f"{role.capitalize()} messages:").grid(row=row, column=0, sticky=tk.W, pady=5)
                ttk.Label(role_tab, text=f"{count}").grid(row=row, column=1, sticky=tk.W, pady=5)
                row += 1
            
            # Cost estimation tab (very rough estimate)
            cost_tab = ttk.Frame(notebook, padding=10)
            notebook.add(cost_tab, text="Cost Estimate")
            
            ttk.Label(cost_tab, text="Estimated Fine-Tuning Costs", font=("Helvetica", 14, "bold")).grid(row=0, column=0, columnspan=2, pady=10)
            ttk.Label(cost_tab, text="(Very rough estimates based on OpenAI pricing)").grid(row=1, column=0, columnspan=2, pady=5)
            
            # Rough cost estimates based on OpenAI pricing (as of 2023)
            training_cost_gpt35 = (total_tokens / 1000) * 0.008  # $0.008 per 1K tokens for GPT-3.5
            training_cost_gpt4 = (total_tokens / 1000) * 0.03  # $0.03 per 1K tokens for GPT-4 (approximation)
            
            ttk.Label(cost_tab, text=f"GPT-3.5 training cost estimate:").grid(row=2, column=0, sticky=tk.W, pady=5)
            ttk.Label(cost_tab, text=f"${training_cost_gpt35:.2f}").grid(row=2, column=1, sticky=tk.W, pady=5)
            
            ttk.Label(cost_tab, text=f"GPT-4 training cost estimate:").grid(row=3, column=0, sticky=tk.W, pady=5)
            ttk.Label(cost_tab, text=f"${training_cost_gpt4:.2f}").grid(row=3, column=1, sticky=tk.W, pady=5)
            
            ttk.Label(cost_tab, text="Note: Actual costs may vary based on OpenAI's current pricing").grid(row=4, column=0, columnspan=2, pady=10)
            
            # Add a close button
            ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)
            
            # Log the analysis
            self.log_message(f"Dataset analysis complete: {valid_examples} valid examples, {total_tokens:,} estimated tokens")
            
        except Exception as e:
            error_msg = f"Error analyzing dataset: {str(e)}"
            self.log_message(error_msg)
            
            detailed_error = f"{error_msg}\n\n"
            detailed_error += "Possible causes:\n"
            detailed_error += "1. File format issues - ensure it's a proper JSONL file\n"
            detailed_error += "2. File encoding issues - ensure the file is UTF-8 encoded\n"
            detailed_error += "3. JSON parsing errors - check for malformed JSON\n"
            detailed_error += "4. Large file size - very large files may cause memory issues\n"
            
            # Show the error in a scrollable dialog
            self.show_error_report("Dataset Analysis Error", detailed_error)
    
    def show_about(self):
        # Switch to the About tab
        self.notebook.select(4)  # Index 4 is the About tab
            
if __name__ == "__main__":
    print("Starting SylvaFine application...")
    print(f"OpenAI API key: {os.getenv('OPENAI_API_KEY')[:5]}... (truncated for security)")
    
    root = tk.Tk()
    app = SylvaFineGUI(root)
    
    # Load settings if available
    app.load_settings()
    
    # Start the application
    root.mainloop()
