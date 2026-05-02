"""
create_dataset.py
-----------------
Generates a realistic synthetic social media dataset for sentiment analysis.
Simulates posts from brands like Zomato, Netflix, Flipkart, and more.
"""

import pandas as pd
import random
import csv
from datetime import datetime, timedelta

random.seed(42)

# ──────────────────────────────────────────────
# 1.  SAMPLE POSTS PER SENTIMENT
# ──────────────────────────────────────────────

POSITIVE_POSTS = [
    "Absolutely love this product! Best purchase I've made all year.",
    "Zomato delivered in 20 minutes. Food was piping hot. 5 stars!",
    "Netflix's new series is absolutely mind-blowing. Can't stop watching!",
    "Flipkart customer service resolved my issue in minutes. Impressed!",
    "The new update is so smooth. Great work by the dev team!",
    "Ordered groceries on Swiggy Instamart, arrived in 8 minutes. Wow!",
    "Amazon Prime delivery was super fast. Product quality is excellent.",
    "Best banking app ever. Zero crashes, super fast transactions.",
    "The movie was fantastic. Brilliant screenplay and acting.",
    "Very happy with my purchase. Packaging was neat and delivery was early.",
    "Exceeded all expectations. Highly recommend to everyone!",
    "Customer support was extremely helpful and resolved everything quickly.",
    "This app makes my life so much easier. Love the UI design.",
    "Just received my order — looks exactly like the photos. Very satisfied!",
    "Great experience from start to finish. Will definitely buy again.",
    "Unboxing was a delight. Premium quality, fast delivery. Worth every rupee.",
    "The new season is brilliant. Netflix knows how to keep us hooked!",
    "Discounts during the sale were amazing. Saved so much money!",
    "Smooth checkout, accurate tracking, on-time delivery. Perfect!",
    "Love how the brand listens to customer feedback. Keep it up!",
    "Finally a food delivery app that gets it right. Bravo Zomato!",
    "The product quality has improved drastically. Really happy!",
    "So easy to use. Even my grandmother figured it out in seconds.",
    "Prompt refund processed in 2 days. That's what I call service!",
    "Brilliant feature update. This is exactly what users needed.",
    "Loving every episode. This is award-winning content for sure.",
    "My go-to shopping app. Never disappointed me once.",
    "The packaging is eco-friendly AND beautiful. Good job brand!",
    "Fast internet speeds all day. The telecom finally delivered.",
    "Swiggy support gave me a refund instantly. Highly impressed.",
]

NEGATIVE_POSTS = [
    "Waited 2 hours for my order and it still hasn't arrived. Terrible!",
    "The food was cold and stale. Never ordering from here again.",
    "Netflix keeps buffering. Your streaming quality is horrible.",
    "Flipkart sent me a broken product. Absolute waste of money.",
    "The app crashes every time I open it. Fix your bugs!",
    "Customer support is useless. They just give scripted replies.",
    "Worst experience ever. Filed a complaint but got no response.",
    "The battery drains so fast on this phone. Very disappointed.",
    "Swiggy cancelled my order with no reason and no refund!",
    "Zomato driver was rude and delivered the wrong order.",
    "Hidden charges appeared at checkout. Totally misleading pricing.",
    "My package has been stuck in transit for 10 days. Unacceptable!",
    "The new update ruined everything. Please roll it back.",
    "Subscription auto-renewed without my consent. Scam!",
    "Terrible UI, hard to navigate, laggy. Not worth using.",
    "Paid for express delivery and it took 5 days. Ridiculous!",
    "Product doesn't match the description at all. Misleading.",
    "The food quality has gone downhill. Not what it used to be.",
    "No one from support has called back in 3 days. Pathetic.",
    "The refund promised 7 days ago has still not arrived.",
    "Constant outages. Your service reliability is a joke.",
    "Overpriced for the quality you get. Not worth it at all.",
    "App UI is confusing and hard to use. Very poor design.",
    "Damaged product delivered. Photos clearly show the damage.",
    "My account got locked for no reason. Very frustrating!",
    "Charged twice for a single order and no one is helping.",
    "Platform is full of fake reviews. Very misleading.",
    "Food arrived without utensils. How do you expect me to eat?",
    "Data privacy is a concern. This app accesses too much.",
    "Three cancellations in a row. Your service is unreliable.",
]

NEUTRAL_POSTS = [
    "Just received my order. Will update once I try the product.",
    "The new Netflix show premiered today. Haven't watched yet.",
    "Zomato has added new restaurants to my area.",
    "Flipkart's sale starts tomorrow. Checking out the deals.",
    "The app just got an update. Downloading it now.",
    "Customer support asked for more details about my issue.",
    "Ordered the product last night. Delivery expected in 3 days.",
    "The new feature is available. Yet to explore it fully.",
    "My subscription renews next week. Checking plans.",
    "Tracking shows my parcel is in Bangalore warehouse.",
    "Just signed up for the service. Will post a review later.",
    "The restaurant I usually order from is closed today.",
    "Waiting for the sale to start to make my purchase.",
    "Saw an ad for this product. Anyone tried it?",
    "My order status shows 'out for delivery'. Let's see.",
    "Read mixed reviews online. Deciding whether to buy.",
    "The cashback will reflect in 3-5 business days apparently.",
    "Notification says my order is nearby. Fingers crossed.",
    "Not sure if I should upgrade to the premium plan.",
    "Saw the trailer. Might watch the new series this weekend.",
    "The price dropped slightly from last week.",
    "Delivery partner just called to confirm my address.",
    "The app asked me to rate my experience. Will do later.",
    "Considering switching plans. Need to compare options first.",
    "Got an OTP for my account. All seems normal.",
    "The new menu options look interesting. Will try next time.",
    "Downloaded the app. Setting up my profile right now.",
    "Saw the movie listing. Might go this weekend.",
    "The email confirmation came through. Waiting now.",
    "Checked the website. The product is still in stock.",
]

BRANDS = ["Zomato", "Swiggy", "Netflix", "Flipkart", "Amazon", "HDFC Bank",
          "Jio", "Airtel", "BookMyShow", "MakeMyTrip", "Ola", "Uber",
          "BYJU's", "Meesho", "Nykaa"]

PLATFORMS = ["Twitter", "Facebook", "Instagram", "Reddit", "App Store",
             "Play Store", "Google Reviews", "Trustpilot"]

LOCATIONS = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
             "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Kochi"]


def random_date(days_back=90):
    base = datetime.now()
    delta = timedelta(days=random.randint(0, days_back),
                      hours=random.randint(0, 23),
                      minutes=random.randint(0, 59))
    return (base - delta).strftime("%Y-%m-%d %H:%M:%S")


def add_noise(text):
    """Occasionally add realistic social media noise."""
    noises = ["#trending", "!!!", "...", "🔥", "👎", "lol", "tbh", "omg",
              "#disappointed", "#happy", "#mustwatch", "#avoid"]
    if random.random() < 0.3:
        text += " " + random.choice(noises)
    return text


def build_dataset(n_per_class=500):
    rows = []
    for label, pool in [("positive", POSITIVE_POSTS),
                        ("negative", NEGATIVE_POSTS),
                        ("neutral",  NEUTRAL_POSTS)]:
        for _ in range(n_per_class):
            text = add_noise(random.choice(pool))
            rows.append({
                "id":         len(rows) + 1,
                "text":       text,
                "sentiment":  label,
                "brand":      random.choice(BRANDS),
                "platform":   random.choice(PLATFORMS),
                "location":   random.choice(LOCATIONS),
                "likes":      random.randint(0, 5000),
                "retweets":   random.randint(0, 1000),
                "timestamp":  random_date(),
            })

    random.shuffle(rows)
    df = pd.DataFrame(rows)
    df.to_csv("data/social_media_posts.csv", index=False)
    print(f"✅  Dataset saved → data/social_media_posts.csv  ({len(df)} rows)")
    print(df["sentiment"].value_counts())
    return df


if __name__ == "__main__":
    build_dataset(n_per_class=500)