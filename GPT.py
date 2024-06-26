import openai

def gpt(object):
    #openai.api_key = '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'

    input('If you want to pick up an' + object + ', which part makes the most sense to grasp? Name one part.')

    msg = {'role':'user', 'content': "If you want to pick up an " + object + ", which part makes the most sense to grasp? Name one part."}
    result = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=[msg])
    answer = result.choices[0].message['content']
    print('grasp part:', answer)
    
    return answer

# gpt("ice cream")
