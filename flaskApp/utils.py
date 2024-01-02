from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf

if __name__=='main':
    # Encode labels
    label_mapping = {'Negative emotion': 0, 'Positive emotion': 1, 'No emotion toward brand or product': 2}
    # df['label'] = df['is_there_an_emotion_directed_at_a_brand_or_product'].map(label_mapping)

    # Load the model and tokenizer
    model = TFBertForSequenceClassification.from_pretrained('fine_tuned_bert_sentiment_model')
    tokenizer = BertTokenizer.from_pretrained('fine_tuned_bert_sentiment_model')


    # Tokenizer
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    max_length = 128

    # New tweets
    new_tweets = [
        "#SXSW is just starting, #CTIA is around the corner and #googleio is only a hop skip and a jump from there, good time to be an #android fan",
        "Beautifully smart and simple idea RT @madebymany @thenextweb wrote about our #hollergram iPad app for #sxsw! http://bit.ly/ieaVOB"
    ]

    # Tokenize and pad the new tweets
    new_input_ids = []
    new_attention_masks = []

    for text in new_tweets:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='tf'
        )

        new_input_ids.append(encoded_dict['input_ids'])
        new_attention_masks.append(encoded_dict['attention_mask'])

    new_input_ids = tf.concat(new_input_ids, axis=0)
    new_attention_masks = tf.concat(new_attention_masks, axis=0)

    # Make predictions
    predictions = model.predict([new_input_ids, new_attention_masks])

    # Convert logits to probabilities
    probabilities = tf.nn.softmax(predictions.logits, axis=-1)

    # Get predicted labels
    predicted_labels = tf.argmax(probabilities, axis=-1).numpy()

    # Map predicted labels back to emotions
    label_mapping_inverse = {v: k for k, v in label_mapping.items()}

    predicted_emotions = [label_mapping_inverse[label] for label in predicted_labels]

    # Display results
    for tweet, emotion in zip(new_tweets, predicted_emotions):
        print(f"Tweet: {tweet}")
        print(f"Predicted Emotion: {emotion}")
        print()