# Bing Spell Check API

Author: Laura Kremers

## TL;DR

Bing Spell Check, an API service under Microsoft Azure, offers a robust solution for grammar and spell checking in texts. This dynamic API goes beyond traditional methods, leveraging advanced linguistic algorithms and machine learning to provide intelligent suggestions for text improvement.

## Key Features

- **Grammar and Spelling Error Recognition**: Bing Spell Check excels in identifying and correcting grammar and spelling errors in written content.
- **Slang Detection and Correction**: It can discern slang usage and provides suggestions for improvement, ensuring that informal language is appropriately refined.
- **Context-Aware Homonym Recognition**: Recognizing words that sound similar but have different meanings, Bing Spell Check suggests contextually relevant corrections for improved accuracy.
- **Brand and Popular Term Recognition**: The API is adept at identifying brands and popular terms, ensuring that specific and widely used terms are recognized and handled appropriately.

## Evolution Beyond Traditional Approaches

While traditional spell-checking methods, such as those employed by Microsoft Word, rely on dictionaries and rule-based systems, Bing Spell Check adopts a revolutionary approach. Traditional methods may mark words not found in the dictionary or those not conforming to predefined rules as potential errors, requiring constant dictionary updates.

In contrast, Bing Spell Check employs a dynamic AI system. Microsoft's innovative approach involves harnessing machine learning and statistical machine translations to train its algorithm. This not only enables the system to adapt to evolving language but also enhances its ability to understand context and offer more accurate suggestions.

The traditional reliance on static dictionaries is replaced by a dynamic, learning algorithm, making Bing Spell Check a cutting-edge solution for anyone seeking to elevate the quality of written communication. Through its multifaceted features, it emerges as a valuable tool in the realm of text correction and enhancement.

## Using the API

1.  Sign Up for Azure and Create a Bing Spell Check Resource:

    - Visit the Azure portal.
    - Sign in or create a new account.
    - In the Azure portal, navigate to "Create a resource" and search for "Bing Spell Check."
    - Create a new Bing Spell Check resource and configure the settings.

2.  Get Your API Key:

    - Once the resource is created, go to the resource dashboard.
    - Find and copy the API key associated with your Bing Spell Check resource.

3.  Construct API Request:

    - Formulate HTTP requests to the Bing Spell Check API endpoint, incorporating your API key.
    - Example API endpoint: https://api.bing.microsoft.com/v7.0/spellcheck

4.  Make API Requests:

    - Use the GET method to send text for spell-checking.
    - Include the text in the request body and set the Ocp-Apim-Subscription-Key header with your API key.

5.  Handle API Responses:

    - The API will respond with spell-checking suggestions.
    - Parse the response to extract suggestions and implement logic to apply corrections as needed.

### Example Code

```python
import requests

# Define the text to be spell-checked
example_text = "arti cle"

def bing_spell_check(text):
    # Replace with your subscription key and endpoint
    subscription_key = ""
    endpoint = "https://api.bing.microsoft.com/v7.0/spellcheck"

    # Set parameters for the API request
    params = {
        'mode': 'spell',
        'text': text
    }

    # Set headers with the subscription key
    headers = {
        "Ocp-Apim-Subscription-Key": subscription_key
    }

    # Make the API request
    response = requests.get(endpoint, headers=headers, params=params)
    response.raise_for_status()

    # Parse the API response
    search_results = response.json()
    corrected_text = text
    flagged_tokens = search_results.get('flaggedTokens', [])
    offset_shift = 0

    # Iterate through flagged tokens and apply corrections
    for token in flagged_tokens:
        offset = token.get('offset') + offset_shift
        token_text = token.get('token')
        suggestions = token.get('suggestions', [])

        if suggestions:
            corrected_token = suggestions[0].get('suggestion')
            # Apply correction to the text
            corrected_text = (
                corrected_text[:offset] + corrected_token + corrected_text[offset + len(token_text):]
            )
            offset_shift += len(corrected_token) - len(token_text)

    return corrected_text

# Call the spell check function and get the corrected text
corrected_text = bing_spell_check(example_text)

# Display the results
print("Original Text:", example_text)
print("Corrected Text:", corrected_text)
```

### Mandatory Query Parameter

- `text`: This parameter is obligatory and represents the text that you want to subject to the spell and grammar check.

### Optional Query Parameters

- `mkt` (Market and Language): This optional parameter allows users to specify the market, which should be in the form of `<language>-<country/region>`. For instance, `en-US` signifies English in the United States.
- `mode` (Check Type): The mode parameter enables users to define the type of spell and grammar check. Two distinct modes are available:
  - `proof` Mode: Offers a comprehensive check, including capitalization and punctuation. Limited to English, Spanish, and Portuguese languages. Restricted to a maximum of 4096 characters.
  - `spell` Mode: Excels in identifying spelling errors, though it may not catch all grammar mistakes. Available for all supported languages.

### Returned Information

Upon making a request to the Bing Spell Check API, the response includes valuable information to assist users in improving their text. Here are key elements of the response:

- `suggestions`: A list of words that correct the spelling or grammar error.
- `suggestion`: The suggested word to replace the flagged word with.
- `token`: The word in the text query string that is not spelled correctly or is grammatically incorrect.
- `type`: The type of error that caused the word to be flagged. Possible values include:
  - `RepeatedToken`
  - `UnknownToken`: All other spelling or grammar errors.
- Score: A value that indicates the level of confidence that the suggested correction is correct.

## Key Takeaways

Bing Spell Check API, part of Microsoft Azure, revolutionizes text correction with advanced linguistic algorithms and machine learning, offering intelligent suggestions for grammar and spelling improvements.

Beyond Traditional Approaches, Bing Spell Check distinguishes itself by replacing static dictionaries with a dynamic AI system, adapting to evolving language nuances and providing context-aware corrections.

Integrating the API is straightforward—sign up for Azure, create a Bing Spell Check resource, obtain an API key, and construct HTTP requests. The API's adaptive responses include valuable insights for enhancing text quality.

## References

- <https://learn.microsoft.com/en-us/previous-versions/azure/cognitive-services/Bing-Spell-Check/quickstarts/python>
- <https://www.microsoft.com/en-us/bing/apis/bing-spell-check-api>
- <https://www.youtube.com/watch?v=-zuqfJaxT8A>
