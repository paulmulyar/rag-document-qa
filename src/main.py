import os
from dotenv import load_dotenv
from anthropic import Anthropic

# Load environment variables
load_dotenv()

def test_setup():
    """Test that everything is set up correctly"""
    
    print("✓ Python is working")
    print(f"✓ Running from: {os.getcwd()}")
    
    # Test API key is loaded
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        print("✓ API key loaded")
        
        # Test API connection
        try:
            client = Anthropic(api_key=api_key)
            message = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=50,
                messages=[{"role": "user", "content": "Say hello!"}]
            )
            print(f"✓ API works! Claude says: {message.content[0].text}")
        except Exception as e:
            print(f"✗ API error: {e}")
    else:
        print("✗ No API key found - check your .env file")

if __name__ == "__main__":
    test_setup()