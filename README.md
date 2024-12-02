# AI-Powered Prompt Chaining System

## Description

This repository contains an advanced AI-powered prompt chaining system that leverages Azure OpenAI's GPT models to generate comprehensive responses to complex queries. The system employs a two-stage approach: a planning stage using a smaller model for task decomposition, followed by a detailed output generation stage using a more powerful model.

## Key Features

- **Two-Stage Processing**: Utilizes gpt-4o-mini for planning and gpt-4o for final output generation.
- **Robust Error Handling**: Implements custom exceptions and retry mechanisms for improved reliability.
- **Configurable Settings**: Centralized configuration management for easy parameter adjustments.
- **Input and Output Validation**: Ensures high-quality responses through thorough validation and quality assessment.
- **Performance Tracking**: Detailed execution statistics for monitoring and optimization.
- **Modular Design**: Well-structured code for easy maintenance and extensibility.

## Requirements

- Python 3.7+
- Azure OpenAI API access
- Required Python packages: `openai`, `python-dotenv`

## Setup

1. Clone the repository:

   ```
   git clone https://github.com/terilios/reasoning-iteration.git
   ```

2. Install required packages:

   ```
   pip install openai python-dotenv
   ```

3. Create a `.env` file in the root directory and add your Azure OpenAI credentials:
   ```
   AZURE_OPENAI_API_KEY=your_api_key
   AZURE_OPENAI_API_BASE=your_api_base
   AZURE_OPENAI_API_VERSION=your_api_version
   AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
   AZURE_OPENAI_MINI_DEPLOYMENT_NAME=your_mini_deployment_name
   ```
   Note: The `.env` file is included in `.gitignore` to prevent sensitive information from being uploaded to the repository.

## Usage

Run the script with a sample prompt:

```python
python main.py
```

Modify the `user_prompt` in the `__main__` section to test different queries.

## Contributing

Contributions to enhance the functionality or efficiency of the system are welcome. Please submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Tags

#AI #MachineLearning #NLP #AzureOpenAI #PromptEngineering #Python

---

This project demonstrates advanced prompt engineering techniques and can serve as a foundation for building sophisticated AI-powered applications.
