import openai

def gpt(object):

    input('If you want to pick up an' + object + ', which part makes the most sense to grasp? Name one part.')

    msg = {'role':'user', 'content': "If you want to pick up an " + object + ", which part makes the most sense to grasp? Name one part."}
    result = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=[msg])
    answer = result.choices[0].message['content']
    print('grasp part:', answer)
    
    return answer

# gpt("ice cream")
