# Contributing to SentiScope AI

Thank you for considering contributing to SentiScope AI! This document provides guidelines and instructions for contributing to the project.

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Screenshots if applicable
- Your environment (OS, Python version, etc.)

### Suggesting Features

Feature suggestions are welcome! Please:
- Check if the feature already exists or is planned
- Describe the feature clearly
- Explain why it would be useful
- Provide examples if possible

### Code Contributions

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/YourFeatureName
   ```
3. **Make your changes**
   - Write clean, readable code
   - Follow PEP 8 style guidelines
   - Add comments where necessary
   - Update documentation

4. **Test your changes**
   - Ensure all existing tests pass
   - Add new tests for new features
   - Test on different browsers/environments

5. **Commit your changes**
   ```bash
   git commit -m "Add: Your feature description"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/YourFeatureName
   ```

7. **Create a Pull Request**
   - Provide clear description of changes
   - Reference any related issues
   - Explain testing performed

## 📋 Development Guidelines

### Code Style

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Keep functions small and focused
- Add docstrings to functions and classes

Example:
```python
def analyze_sentiment(text: str) -> dict:
    """
    Analyze sentiment of given text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary with sentiment and confidence scores
    """
    # Implementation here
    pass
```

### Commit Messages

Use clear, descriptive commit messages:
- `Add: New feature description`
- `Fix: Bug description`
- `Update: Changes to existing feature`
- `Docs: Documentation updates`
- `Style: Code formatting changes`
- `Refactor: Code restructuring`

### Testing

- Test all changes locally before submitting
- Ensure the app runs without errors
- Test on multiple screen sizes (desktop, tablet, mobile)
- Verify all features work as expected

## 🏗️ Project Structure

```
SentiScope-AI/
├── streamlit_app.py      # Main application
├── src/
│   └── compare_models.py # Utilities
├── models/               # ML models
├── data/                 # Sample data
└── outputs/              # Results
```

## 🔧 Setting Up Development Environment

1. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/ReviewLens.git
   cd ReviewLens
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run locally**
   ```bash
   streamlit run streamlit_app.py
   ```

## 📝 Documentation

When adding features:
- Update README.md if needed
- Add docstrings to functions
- Include usage examples
- Update this CONTRIBUTING guide if needed

## ✅ Pull Request Checklist

Before submitting your PR, ensure:
- [ ] Code follows PEP 8 style guidelines
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] No unnecessary files included
- [ ] Feature works on different screen sizes
- [ ] No console errors

## 🐛 Bug Fix Process

1. Create an issue describing the bug
2. Fork and create a branch named `fix/bug-description`
3. Fix the bug
4. Add tests to prevent regression
5. Submit PR referencing the issue

## 💡 Feature Request Process

1. Open an issue with the feature request
2. Discuss the feature with maintainers
3. Get approval before starting work
4. Implement the feature
5. Submit PR with comprehensive description

## 🎨 UI/UX Contributions

For design changes:
- Maintain consistent color scheme
- Ensure responsive design
- Test on different browsers
- Provide screenshots in PR
- Explain design decisions

## 📊 Adding ML Models

If contributing new models:
- Document model architecture
- Include training data requirements
- Provide performance metrics
- Add model comparison results
- Update documentation

## 🌍 Internationalization

For adding language support:
- Create language files in proper format
- Translate all UI strings
- Test with sample data in that language
- Update language selection UI

## 🚀 Performance Optimization

Performance improvements welcome:
- Profile code to identify bottlenecks
- Optimize without breaking functionality
- Include before/after benchmarks
- Document optimization approach

## 📧 Contact

Questions about contributing?
- Open a discussion on GitHub
- Email: shishirkawa14@gmail.com

## 🙏 Thank You!

Thank you for contributing to SentiScope AI and helping make sentiment analysis more accessible!

---

**Built with ❤️ by Shishir Kawa**
