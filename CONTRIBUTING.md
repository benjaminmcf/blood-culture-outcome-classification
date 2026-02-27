# Contributing to blood-culture-outcome-classification

Thank you for your interest in contributing to this project! This document provides guidelines for contributions.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on GitHub with:

1. A clear, descriptive title
2. Steps to reproduce the issue
3. Expected vs actual behaviour
4. Your environment (Python version, OS, package versions)

### Suggesting Enhancements

Enhancement suggestions are welcome. Please open an issue describing:

1. The use case or problem you're trying to solve
2. Your proposed solution
3. Any alternatives you've considered

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Add or update tests as appropriate
5. Ensure all tests pass (`pytest tests/ -v`)
6. Commit with clear messages
7. Push to your fork and open a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/blood-culture-outcome-classification.git
cd blood-culture-outcome-classification

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies including dev tools
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Add docstrings to public functions
- Keep functions focused and modular

## Testing

- Add tests for new functionality
- Ensure existing tests pass before submitting PR
- Aim for meaningful coverage of edge cases

## Documentation

- Update README.md if adding new features
- Add docstrings to new functions
- Update the paper/manuscript if relevant to publication

## Questions?

Feel free to open an issue for any questions about contributing.

## Code of Conduct

Please be respectful and constructive in all interactions. We are committed to providing a welcoming environment for all contributors.
