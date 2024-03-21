# **SentimentScope**: Flipkart Review Insight Engine

## Overview
**SentimentScope** is a web application designed to analyze and provide insights into Flipkart product reviews. It employs sentiment analysis, named entity recognition (NER), topic modeling, and visualization techniques to help users understand customer sentiments and identify key topics from the reviews.

The application allows users to input review texts manually, upload files containing review data, or enter Flipkart product URLs to scrape and analyze reviews directly from the website. Users can customize the analysis by selecting languages, specifying aspect keywords, and providing custom stopwords.

## Features
- **Sentiment Analysis:** Predicts the sentiment of reviews as positive or negative using a pre-trained machine learning model.
- **Named Entity Recognition (NER):** Identifies entities such as product names, organizations, and locations mentioned in the reviews.
- **Topic Modeling:** Extracts key topics from the reviews using Latent Dirichlet Allocation (LDA) and visualizes them for better understanding.
- **Visualization:** Generates visualizations including bar charts, pie charts, histograms, scatter plots, and word clouds to represent sentiment distribution, review length distribution, sentiment over time, and more.
- **Multi-Language Support:** Supports multiple languages for review analysis and translation using the Google Translate API.
- **Customization:** Allows users to filter reviews by language, specify aspect keywords, remove custom stopwords, and visualize sentiment intensity.

## Installation
1. Clone the repository from GitHub:
    ```bash
    git clone https://github.com/revanthchristober/SentimentScope.git
    ```

2. Navigate to the project directory:
    ```bash
    cd SentimentScope
    ```

3. Install the required dependencies using pip:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

5. Access the application in your web browser at **Network URL** provided by **streamlit**.

## Usage
1. Launch the application by running the Streamlit command in your terminal.
2. Choose the input type: manual entry, file upload, or Flipkart product URL (*works when if it's running from local machine*).
3. Customize the analysis by selecting languages, specifying aspect keywords, or providing custom stopwords.
4. Analyze the sentiment, entities, and topics of the reviews.
5. Visualize the review insights using various charts and graphs.
6. Download the analyzed data or share the visualizations as needed.

## Contributing
Contributions to **SentimentScope** are welcome! If you find any issues or have suggestions for improvements, please open an issue or create a pull request on GitHub.

## License
This project is licensed under the GNU 3.0 License - see the [LICENSE](LICENSE) file for details.

### Contact
For questions, feedback, or inquiries, please contact [revanthchrixtopher@outlook.com](mailto:revanthchrixtopher@outlook.com).
---
