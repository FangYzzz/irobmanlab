import sys
sys.path.append('/home/yuan/Mani-GPT/Detic')

import io
from PIL import Image
from matplotlib import pyplot as plt

import openai
import Detic.demo
from owl_vit_minimal_example import detect_grasp_part



def get_grasp_object(assistant_message):
    grasp_object = assistant_message.split('Here is your ')[1].split('.')[0].strip()
    print(f"(Grasp object: {grasp_object})")

    return grasp_object


def gpt_grasp_part(grasp_object):
    openai.api_key = ''
    # input("If you want to pick up an {grasp_object}, which part makes the most sense to grasp? Name one part.")

    msg = {'role':'user', 'content': "If you want to pick up " + grasp_object + ", which part makes the most sense to grasp? Name one part(just a phrase)."}
    result = openai.ChatCompletion.create(model='gpt-4', messages=[msg])      # gpt-3.5-turbo
    grasp_part = result.choices[0].message['content']
    print(f"(Grasp part: {grasp_part})")
    
    return grasp_part



def get_chatgpt_response(messages):
    try:
        response = openai.ChatCompletion.create(
            model ="gpt-4",
            messages = messages,
            max_tokens = 150
        )

        assistant_message = response.choices[0].message['content'].strip()    
        return assistant_message

    except Exception as e:
        return f"error: {str(e)}"



def gpt_dialogue():
    openai.api_key = ''
    
    message = [
    {"role": "system", "content": """
        Task: You are AI and in a kitchen. I am Human and will ask you for help. I will tell you, the objects on the table. Generate answer for object search and grasp. 
        You need to answer Action and Response. Actions are classified into four categories: confirm, respond, refuse and grasp. In grasp action, the response always is "Here is your + object name".
        Hier is a prompt for the task below:
        Start: you can see egg, saucepan, knife, bell pepper on the table.
        Human: Hi
        AI: Action: <respond>
            Response: Hello! What can I do for you?
        Human: I want to make a meal and I need some vegatables.
        AI: Action:  <confirm>
            Response: I can see Bell pepper. Would you like me to bring it to you?
        Human: Sure
        AI: Action: <grasp>
            Response: Here is your bell pepper.
        Human: Give me the milk please, I'm so thirsty.
        AI: Action: <respond>
            Response: I'm sorry. I cannot see any milk on the table. 
        Human: OK, bring me the knife, my kid wants to play.
        AI: Action: <refuse>
            Response: I'm sorry, but it is not safe for a kid to play knives. May be we can find something else for fun.
        Human: You are right. I should keep him away from this. Thank you.
        AI: Action: <respond>
        Response:You are welcome. Call me any time when you need another help.
        """}
        ]
    response_1 = get_chatgpt_response(message)
    # print(f"ChatGPT1: {response_1}")
    message.append({"role": "assistant", "content": response_1})
    

    ### demo output ###
    exist_object_prompt = Detic.demo.detect_exist_objects()
    message.append({"role": "user", "content": exist_object_prompt})

    response_2 = get_chatgpt_response(message)
    message.append({"role": "assistant", "content": response_2})
    # print(f"GPT2: {response_2}")



    input("Type 'q' to end the conversation")
    while True:
        user_input = input("\nHuman: ")
        message.append({"role": "user", "content": user_input})
        if user_input.lower() == 'q':
            print("Conversation ended")
            break

        response = get_chatgpt_response(message)
        message.append({"role": "assistant", "content": response})
        print(f"{response}")
        
        if "<grasp>" in response:
            grasp_object = get_grasp_object(response)
            grasp_part = gpt_grasp_part(grasp_object)

            # convert a byte stream to an image object
            image_bytes = detect_grasp_part(grasp_part)
            image = Image.open(io.BytesIO(image_bytes))

            # show the image
            plt.imshow(image)
            plt.axis('off')        # Hide axes
            plt.show()

        


gpt_dialogue()





# def get_chatgpt_response(messages):
#     try:
#         response = openai.ChatCompletion.create(
#             model ="gpt-4",
#             messages = messages,
#             max_tokens = 1000
#         )

#         assistant_message = response.choices[0].message['content'].strip()
#         if "<grasp>" in assistant_message:
#             grasp_object = get_grasp_object(assistant_message)

#             grasp_part = gpt_grasp_part(grasp_object)

#             return grasp_part, assistant_message
#         else:
#             return None, assistant_message

#     except Exception as e:
#         return f"error: {str(e)}"