import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')
nltk.download('vader_lexicon', quiet=True)

st.set_page_config(
    page_title='Apple Stock Predictor',
    page_icon='📈',
    layout='wide'
)

@st.cache_resource
def load_everything():

    model = load_model('best_model.keras', compile=False)
    model.compile(optimizer='adam', loss='mse')

    with open('feature_scaler.pkl', 'rb') as f:
        feature_scaler = pickle.load(f)

    with open('target_scaler.pkl', 'rb') as f:
        target_scaler = pickle.load(f)

    # Load the feature column names
    with open('feature_columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)

    # Load the final dataset (used for last 60 days of features)
    df = pd.read_csv('final_dataset.csv')
    df['date'] = pd.to_datetime(df['date'])

    return model, feature_scaler, target_scaler, feature_columns, df

model, feature_scaler, target_scaler, feature_columns, df = load_everything()
sia = SentimentIntensityAnalyzer()

def get_sentiment(headline):
    scores = sia.polarity_scores(str(headline))
    return scores['compound']

def assign_impact_tier(score):
    if score >= 0.5 or score <= -0.5:
        return 'HIGH'
    elif score > 0 or score < 0:
        return 'MEDIUM'
    else:
        return 'LOW'

def get_impact_color(tier):
    if tier == 'HIGH':
        return 'red' if True else 'green'
    elif tier == 'MEDIUM':
        return 'orange'
    else:
        return 'gray'

def predict_tomorrow(headline, df, model, feature_scaler, target_scaler, feature_columns):

    # Step 1: Get sentiment from headline using VADER
    sentiment_score = get_sentiment(headline)
    impact_tier     = assign_impact_tier(sentiment_score)

    # Step 2: Take last 60 rows of the dataset as our base window
    last_60 = df[feature_columns].tail(60).copy()

    # Step 3: Reset index so it goes from 0 to 59
    # This is the key fix - ensures .at works correctly
    last_60 = last_60.reset_index(drop=True)

    # Step 4: Update the last row's news sentiment features
    # using the user's headline sentiment
    # We use .at[row_number, column_name] which is safe and simple

    if 'avg_sentiment' in last_60.columns:
        last_60.at[59, 'avg_sentiment'] = sentiment_score

    if 'max_sentiment' in last_60.columns:
        current_max = last_60.at[59, 'max_sentiment']
        last_60.at[59, 'max_sentiment'] = max(sentiment_score, current_max)

    if 'min_sentiment' in last_60.columns:
        current_min = last_60.at[59, 'min_sentiment']
        last_60.at[59, 'min_sentiment'] = min(sentiment_score, current_min)

    if 'high_impact_count' in last_60.columns and impact_tier == 'HIGH':
        last_60.at[59, 'high_impact_count'] += 1

    if 'medium_impact_count' in last_60.columns and impact_tier == 'MEDIUM':
        last_60.at[59, 'medium_impact_count'] += 1

    if 'negative_count' in last_60.columns and sentiment_score < 0:
        last_60.at[59, 'negative_count'] += 1

    if 'news_count' in last_60.columns:
        last_60.at[59, 'news_count'] += 1

    # Step 5: Scale the 60-day window using the feature scaler
    scaled_window = feature_scaler.transform(last_60.values)

    # Step 6: Reshape for LSTM input
    # LSTM expects shape (samples, window_size, features)
    lstm_input = scaled_window.reshape(1, 60, len(feature_columns))

    # Step 7: Make prediction
    prediction_scaled = model.predict(lstm_input, verbose=0)

    # Step 8: Convert back to real dollar price
    prediction_real = target_scaler.inverse_transform(prediction_scaled)

    return prediction_real[0][0], sentiment_score, impact_tier

# ============================================================
# Streamlit UI
# ============================================================

# Header
st.title('📈 Apple Stock Price Predictor')
st.markdown('**Powered by LSTM Neural Network + News Sentiment Analysis**')
st.markdown('---')

# Two columns layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader('Enter Today\'s News Headline')

    # Text input for news headline
    headline = st.text_area(
        label='Type a news headline about Apple or the market:',
        placeholder='Example: Apple reports record breaking quarterly earnings...',
        height=100
    )

    # Predict button
    predict_button = st.button('🔮 Predict Tomorrow\'s Price', use_container_width=True)

with col2:
    st.subheader('Recent Apple Stock Prices')

    # Show last 10 days of closing prices
    recent = df[['date', 'Close']].tail(10).copy()
    recent['date'] = recent['date'].dt.strftime('%Y-%m-%d')
    recent = recent.rename(columns={'Close': 'Close Price ($)'})
    recent = recent.reset_index(drop=True)
    st.dataframe(recent, use_container_width=True)

# ============================================================
# When button is clicked
# ============================================================
st.markdown('---')

if predict_button:

    # Check if headline is empty
    if headline.strip() == '':
        st.warning('Please enter a news headline before clicking Predict.')

    else:
        # Make prediction
        with st.spinner('Analyzing sentiment and predicting price...'):
            predicted_price, sentiment_score, impact_tier = predict_tomorrow(
                headline, df, model, feature_scaler, target_scaler, feature_columns
            )

        # Show results in 3 columns
        st.subheader('Prediction Results')
        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric(
                label='Sentiment Score',
                value=round(sentiment_score, 4),
                help='Score from -1 (very negative) to +1 (very positive)'
            )

        with r2:
            st.metric(
                label='Impact Tier',
                value=impact_tier,
                help='HIGH = strong sentiment, MEDIUM = mild, LOW = neutral'
            )

        with r3:
            # Get last actual close price for comparison
            last_close = df['Close'].iloc[-1]
            price_diff = predicted_price - last_close

            st.metric(
                label='Predicted Tomorrow\'s Close',
                value='$' + str(round(predicted_price, 2)),
                delta=str(round(price_diff, 2)) + ' from today',
                help='Predicted using LSTM model with your news sentiment'
            )

        st.markdown('---')

        # ============================================================
        # Chart: Last 60 days + prediction point
        # ============================================================
        st.subheader('Price Chart: Last 60 Days + Tomorrow\'s Prediction')

        last_60_chart = df[['date', 'Close']].tail(60).copy()

        # Create tomorrow's date
        last_date      = df['date'].iloc[-1]
        tomorrow_date  = last_date + pd.Timedelta(days=1)

        # Skip weekends for the display date
        if tomorrow_date.weekday() == 5:   # Saturday
            tomorrow_date = tomorrow_date + pd.Timedelta(days=2)
        elif tomorrow_date.weekday() == 6: # Sunday
            tomorrow_date = tomorrow_date + pd.Timedelta(days=1)

        fig, ax = plt.subplots(figsize=(12, 5))

        # Plot last 60 days
        ax.plot(last_60_chart['date'], last_60_chart['Close'],
                color='steelblue', linewidth=1.5, label='Actual Close Price')

        # Plot predicted point
        ax.scatter(tomorrow_date, predicted_price,
                   color='red', s=100, zorder=5, label='Predicted Tomorrow')

        # Connect last actual point to prediction with dotted line
        ax.plot([last_60_chart['date'].iloc[-1], tomorrow_date],
                [last_60_chart['Close'].iloc[-1], predicted_price],
                color='red', linestyle='--', linewidth=1.2)

        ax.set_title('Apple Stock - Last 60 Days + Tomorrow Prediction')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price in USD')
        ax.legend()
        fig.tight_layout()

        st.pyplot(fig)

        # ============================================================
        # Sentiment explanation
        # ============================================================
        st.markdown('---')
        st.subheader('How the Headline Influenced the Prediction')

        if sentiment_score > 0.5:
            st.success('The headline has a STRONG POSITIVE sentiment. '
                       'This has pushed the prediction upward.')
        elif sentiment_score > 0:
            st.info('The headline has a MILD POSITIVE sentiment. '
                    'Slight positive influence on prediction.')
        elif sentiment_score == 0:
            st.info('The headline is NEUTRAL. '
                    'No significant sentiment influence on prediction.')
        elif sentiment_score > -0.5:
            st.warning('The headline has a MILD NEGATIVE sentiment. '
                       'Slight downward influence on prediction.')
        else:
            st.error('The headline has a STRONG NEGATIVE sentiment. '
                     'This has pushed the prediction downward.')

st.markdown('---')
st.markdown('Built with LSTM + VADER Sentiment Analysis | BTech CSE 6th Semester | IUST')