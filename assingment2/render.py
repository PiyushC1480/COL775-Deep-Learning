import json



'''
input json contains examples of the form:

{
    "Problem": "<sos>the mean daily profit made by a shopkeeper in a month of 30 days was rs . 350 . if the mean profit for the first fifteen days was rs . 255 , then the mean profit for the last 15 days would be<eos>",
    "answer": "445",
    "predicted": "<sos>multiply(n0,n1)|multiply(n2,n3)|subtract(#0,#1)|divide(#2,n3)|<eos>",
    "linear_formula": "<sos>multiply(n0,n1)|multiply(n2,n3)|subtract(#0,#1)|divide(#2,n3)|<eos>",
    "predicted_answer": null
}

just change the "answer": "445" to "answer": 445 and create a new json with the name : dev_data.json
and remove <sos> and <eos> from the "Problem" field and "predicted" field and "linear_formula" field
'''
def change_answer_field():
    data = json.load(open('test_output.json'))
    for i in range(len(data)):
        data[i]['answer'] = int(data[i]['answer'])
        data[i]['Problem'] = data[i]['Problem'].replace('<sos>', '').replace('<eos>', '')
        data[i]['predicted'] = data[i]['predicted'].replace('<sos>', '').replace('<eos>', '')
        data[i]['linear_formula'] = data[i]['linear_formula'].replace('<sos>', '').replace('<eos>', '')

    with open('dev_data.json', 'w') as f:
        json.dump(data, f, indent='\t', separators=(',', ': '))

if __name__ == '__main__':
    change_answer_field()