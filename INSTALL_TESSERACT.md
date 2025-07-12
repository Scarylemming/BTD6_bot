# Installing Tesseract OCR

This guide will help you install Tesseract OCR, which is required for the round number detection feature.

## Windows Installation

### Method 1: Using the Official Installer (Recommended)

1. **Download Tesseract**:
   - Go to: https://github.com/UB-Mannheim/tesseract/wiki
   - Download the latest installer for your system:
     - `tesseract-ocr-w64-setup-5.3.1.20230401.exe` (64-bit)
     - `tesseract-ocr-w32-setup-5.3.1.20230401.exe` (32-bit)

2. **Install Tesseract**:
   - Run the installer as Administrator
   - **IMPORTANT**: Check "Add to PATH" during installation
   - Choose installation directory (default: `C:\Program Files\Tesseract-OCR`)
   - Complete the installation

3. **Verify Installation**:
   - Open Command Prompt or PowerShell
   - Run: `tesseract --version`
   - You should see version information

### Method 2: Using Chocolatey

```powershell
# Install Chocolatey first if you don't have it
# Then run:
choco install tesseract
```

### Method 3: Using Winget

```powershell
winget install UB-Mannheim.TesseractOCR
```

## macOS Installation

### Method 1: Using Homebrew (Recommended)

```bash
# Install Homebrew first if you don't have it
# Then run:
brew install tesseract
```

### Method 2: Using MacPorts

```bash
sudo port install tesseract
```

### Method 3: Using the Official Installer

1. Download from: https://github.com/tesseract-ocr/tesseract/releases
2. Install the `.dmg` file
3. Add to PATH manually if needed

## Linux Installation

### Ubuntu/Debian

```bash
sudo apt update
sudo apt install tesseract-ocr
sudo apt install tesseract-ocr-eng  # English language pack
```

### CentOS/RHEL/Fedora

```bash
# CentOS/RHEL
sudo yum install tesseract
sudo yum install tesseract-langpack-eng

# Fedora
sudo dnf install tesseract
sudo dnf install tesseract-langpack-eng
```

### Arch Linux

```bash
sudo pacman -S tesseract
sudo pacman -S tesseract-data-eng
```

## Verifying Installation

After installation, verify that Tesseract is working:

```bash
# Check version
tesseract --version

# Test OCR on a simple image
echo "Hello World" > test.txt
tesseract test.txt stdout
```

## Troubleshooting

### If Tesseract is not found in PATH:

1. **Windows**:
   - Find where Tesseract is installed (usually `C:\Program Files\Tesseract-OCR`)
   - Add this directory to your system PATH:
     1. Open System Properties → Advanced → Environment Variables
     2. Edit the PATH variable
     3. Add the Tesseract installation directory
     4. Restart your terminal/IDE

2. **macOS/Linux**:
   - Find Tesseract location: `which tesseract`
   - If not found, add to PATH in your shell profile:
     ```bash
     echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.bashrc
     source ~/.bashrc
     ```

### Using Custom Path in Python

If you can't add Tesseract to PATH, specify the path in your config file:

```python
# In config.py
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Windows
# TESSERACT_PATH = "/usr/bin/tesseract"  # Linux/macOS
```

### Common Issues

1. **"tesseract is not installed or it's not in your PATH"**:
   - Install Tesseract following the steps above
   - Make sure to check "Add to PATH" during Windows installation
   - Restart your terminal/IDE after installation

2. **Permission denied errors**:
   - Run installer as Administrator (Windows)
   - Use `sudo` for package managers (Linux/macOS)

3. **Language pack issues**:
   - Install additional language packs if needed
   - Default English pack should be sufficient for round numbers

## Testing the Installation

After installing Tesseract, test it with the bot:

```bash
# Run the main bot
python main.py

# Or test OCR specifically
python test_round_ocr.py
```

You should see:
```
✅ Tesseract version: 5.3.1.20230401
```

If you see this message, Tesseract is properly installed and the OCR features will work correctly. 