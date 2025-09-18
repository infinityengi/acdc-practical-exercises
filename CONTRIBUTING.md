# ACDC Practical Exercises - Contributing Guidelines

Thank you for your interest in contributing to the ACDC Practical Exercises portfolio! This repository serves as a comprehensive learning resource for autonomous driving technologies.

## 🎯 Purpose and Scope

This repository is designed as a **student portfolio** showcasing practical implementations across key autonomous driving domains:

- Image Segmentation
- Point Cloud Processing  
- Object Tracking
- Occupancy Mapping
- Vehicle Guidance
- V2X Communication

## 🤝 How to Contribute

### Types of Contributions Welcome

1. **Bug Fixes**
   - Corrections to code errors
   - Documentation fixes
   - Broken links or formatting issues

2. **Enhancements**
   - Improved algorithm implementations
   - Better visualization functions
   - Performance optimizations
   - Additional utility functions

3. **Educational Content**
   - New example notebooks
   - Tutorial improvements
   - Additional explanations
   - Real-world case studies

4. **Dataset Integration**
   - Scripts to download/process standard datasets
   - Example data preprocessing pipelines
   - Data format converters

5. **Testing and Validation**
   - Unit tests for core functions
   - Integration tests for modules
   - Performance benchmarks

### What We Don't Accept

- Major architectural changes (this is a learning portfolio)
- Proprietary or copyrighted content
- Large binary files without justification
- Content unrelated to autonomous driving

## 📋 Contribution Process

### 1. Getting Started

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/acdc-practical-exercises.git
cd acdc-practical-exercises

# Create a virtual environment
python -m venv acdc_env
source acdc_env/bin/activate  # On Windows: acdc_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify setup
python setup_check.py
```

### 2. Making Changes

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Make your changes
# ... edit files ...

# Test your changes
python -m pytest tests/  # If tests exist
jupyter notebook  # Verify notebooks work

# Commit your changes
git add .
git commit -m "Add: Brief description of your changes"
```

### 3. Submitting Changes

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create a Pull Request on GitHub
# Include a clear description of your changes
```

## 📝 Coding Standards

### Python Code Style
- Follow PEP 8 conventions
- Use meaningful variable and function names
- Add docstrings for functions and classes
- Keep functions focused and modular

```python
def calculate_iou(pred_mask, true_mask, num_classes=19):
    """
    Calculate Intersection over Union for segmentation masks.
    
    Args:
        pred_mask (np.ndarray): Predicted segmentation mask
        true_mask (np.ndarray): Ground truth mask  
        num_classes (int): Number of segmentation classes
        
    Returns:
        tuple: (class_ious, mean_iou)
    """
    # Implementation here
    pass
```

### Jupyter Notebooks
- Clear markdown explanations for each section
- Executable code cells with meaningful output
- Visualizations with proper labels and titles
- Educational comments explaining key concepts

### Documentation
- Use clear, concise language
- Include practical examples
- Reference academic papers when appropriate
- Maintain consistent formatting

## 📁 Repository Structure

When adding new content, follow the established structure:

```
module_name/
├── README.md                 # Module overview and learning objectives
├── notebooks/               # Jupyter notebooks with examples
│   ├── 01_basics.ipynb
│   └── 02_advanced.ipynb
├── src/                     # Source code modules
│   ├── __init__.py
│   ├── algorithms/
│   ├── utils/
│   └── visualization/
├── data/                    # Sample datasets and test files
├── docs/                    # Additional documentation
└── examples/                # Standalone example scripts
```

## 🧪 Testing Guidelines

### For Code Contributions
- Add unit tests for new functions
- Ensure existing tests still pass
- Include integration tests for complex features
- Test with different Python versions (3.8+)

### For Notebooks
- Ensure all cells execute without errors
- Test with fresh kernel restart
- Verify visualizations display correctly
- Check that examples are educational

## 📚 Educational Focus

Remember this is an **educational repository**. Contributions should:

- Explain concepts clearly
- Provide step-by-step implementations
- Include relevant theory and background
- Show practical applications
- Help students understand autonomous driving

### Good Example:
```python
# Kalman filter prediction step
# This implements the predict phase where we estimate the next state
# based on our motion model and previous state estimate
def predict(self):
    """
    Predict the next state using the motion model.
    
    The prediction step assumes constant velocity motion:
    new_position = old_position + velocity * time_step
    """
    self.x = self.F @ self.x  # State prediction
    self.P = self.F @ self.P @ self.F.T + self.Q  # Covariance prediction
```

## 🎓 Academic Integrity

- Cite sources for algorithms and methods
- Don't copy code without proper attribution
- Acknowledge dataset sources
- Reference relevant academic papers

Example:
```python
"""
Implementation based on:
- "U-Net: Convolutional Networks for Biomedical Image Segmentation" 
  Ronneberger et al., MICCAI 2015
- Original implementation: https://github.com/...
"""
```

## 🐛 Issue Reporting

When reporting issues:

1. **Check existing issues** first
2. **Provide clear description** of the problem
3. **Include steps to reproduce**
4. **Specify your environment** (OS, Python version, etc.)
5. **Add relevant error messages**

### Issue Template:
```
**Description:**
Brief description of the issue

**Steps to Reproduce:**
1. Go to...
2. Run...
3. See error

**Expected Behavior:**
What should happen

**Environment:**
- OS: [e.g., Ubuntu 20.04]
- Python: [e.g., 3.9.7]  
- Dependencies: [versions if relevant]

**Additional Context:**
Any other relevant information
```

## 💬 Communication

- **Be respectful** and constructive
- **Ask questions** if you're unsure
- **Help others** when you can
- **Focus on learning** and education

## 📄 License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers the project.

## 🙏 Recognition

Contributors will be acknowledged in:
- Repository contributors list
- Relevant module documentation
- Release notes (for significant contributions)

## 📞 Getting Help

If you need help:
- Check the module README files
- Look at existing examples  
- Open an issue for questions
- Review the setup verification script

---

**Thank you for helping make autonomous driving education more accessible!** 🚗🤖

Your contributions help students worldwide learn about this exciting field.