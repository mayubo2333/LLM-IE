import backoff
import openai
import asyncio


@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError), max_tries=10)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.acreate(**kwargs)


async def dispatch_openai_requests(messages_list, model, temperature):
    """Dispatches requests to OpenAI API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.
    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        completions_with_backoff(
            model=model,
            messages=x,
            temperature=temperature,
        )for x in messages_list
    ]
    return await asyncio.gather(*async_responses, return_exceptions=True) 