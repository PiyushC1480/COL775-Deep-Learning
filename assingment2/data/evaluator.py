import json 
import sys 
import re 
import math
import pickle
LPAR = '('
RPAR = ')'
ADD = 'add'
SUB = 'subtract'
DIV = 'divide'
POW = 'power'
MUL = 'multiply'
NEG = 'negate'
CAREA = 'circle_area'
SAREA = 'square_area'
RAREA = 'rectangle_area'
VCYL = 'volume_cylinder'
CONST = 'const_'
FLOOR = 'floor'
SQRT = 'sqrt'
CIRCUM = 'circumface'
INV = 'inverse'
FACT = 'factorial'
LOG = 'log'
CHOOSE = 'choose'
RHPERI = 'rhombus_perimeter'
RPERI = 'rectangle_perimeter'
QAREA = 'quadrilateral_area'
SPEED = 'speed'
MOD = 'reminder'
VRPR = 'volume_rectangular_prism'
PERMUT = 'permutation'
ASPHERE = 'surface_sphere'
TAREA = 'triangle_area'
GCD = 'gcd'
LCM = 'lcm'
TPERI = 'triangle_perimeter'
PGAIN = 'p_after_gain'
TAREA3 =  'triangle_area_three_edges'
CUBE_EDGE = 'cube_edge_by_volume'
SRPR = 'surface_rectangular_prism'
SQPERI = 'square_perimeter'
MAX = 'max'
SRCU = 'surface_cube'
VCU = 'volume_cube'
RHAREA =  'rhombus_area'
OPRBLOSS = 'original_price_before_loss'
SQUAE_EDGE =  'square_edge_by_area'
VCONE = 'volume_cone'
SS = 'stream_speed'
SURF_CYL = 'surface_cylinder'
VSPHERE = 'volume_sphere'
OPRBGAIN =  'original_price_before_gain'
MIN = 'min'
SQR_EDGE_BY_PERI = 'square_edge_by_perimeter'
NEG_PROB =  'negate_prob'

#factorial function
def factorial(n):
    ans = 1
    for i in range(1, int(n+1)):
        ans *= i
    return ans 

#lcm function
def lcm(a, b):
    return abs(a * b) // math.gcd(a, b)

#area of triangle
def triangle_area(a, b, c):
    # Calculate the semi-perimeter
    s = (a + b + c) / 2
    
    # Calculate the area using Heron's formula
    area = math.sqrt(s * (s - a) * (s - b) * (s - c))
    return area

#the huge operation dict
ops = {

    ADD: lambda x,y: x+y,
    SUB: lambda x,y: y-x,
    DIV: lambda x,y: y/x,
    POW: lambda x,y: y**x,
    MUL: lambda x,y: x*y,
    NEG: lambda x: -1*x,
    CAREA: lambda x: (x**2)*math.pi,
    VCYL: lambda x,y: 1/3*math.pi*(x**2)*y,
    FLOOR: lambda x: int(x),
    SQRT: lambda x: (x**0.5),
    CIRCUM: lambda x: 2*math.pi*x,
    SAREA: lambda x: x**2,
    INV: lambda x: 1/x,
    RAREA: lambda x,y: x*y,
    FACT: factorial,
    LOG: lambda x: math.log(x),
    CHOOSE: lambda x,y: factorial(y)/(factorial(x)*factorial(y-x)),
    RPERI: lambda x,y: 2*(x+y),
    QAREA: lambda x,y,z: (y+x)/2*z, #new 15141
    SPEED: lambda x,y: y/x,
    MOD: lambda x,y: int(y)%int(x),
    VRPR: lambda x,y,z: y*z*x,
    RHPERI: lambda x: 4*x,
    PERMUT: lambda x,y: factorial(y)/factorial(x),
    ASPHERE: lambda x: 4*math.pi*(x**2),
    TAREA: lambda x,y: (x*y)/2,
    GCD: lambda x,y: math.gcd(int(y),int(x)),
    LCM: lambda x,y: lcm(int(y), int(x)),
    TPERI: lambda x,y,z: (x+y+z),
    PGAIN: lambda x,y: y/x,
    TAREA3: lambda x,y,z: triangle_area(x,y,z),
    CUBE_EDGE: lambda x: x**(1/3),
    SRPR: lambda x,y,z: 2*(x*y + y*z + z*x),
    SQPERI: lambda x: 4*x,
    MAX: lambda x,y: max(x,y),
    SRCU: lambda x: 6*(x**2),
    VCU: lambda x: 6*(x**2),
    RHAREA: lambda x,y: (x+y)/2,
    OPRBLOSS: lambda x,y: (100 - y)/100*x,
    SQUAE_EDGE: lambda x: (x**0.5),
    VCONE: lambda x,y: math.pi*(y**2)/3*y,
    SS: lambda x,y: (x+y)/2,
    SURF_CYL: lambda x,y: math.pi*2*x*y + 2*math.pi*(y**2),
    VSPHERE: lambda x: 4/3*math.pi*(x**3),
    OPRBGAIN: lambda x,y: (100+y)/100*x,
    MIN: lambda x,y: min(x,y),
    SQR_EDGE_BY_PERI: lambda x: x/4,
    NEG_PROB: lambda x: (1-x)

}

#unitary operators
uni_op = [NEG, SQRT, FLOOR, CAREA, CIRCUM, SAREA, INV, 
            FACT, LOG, RHPERI, ASPHERE, CUBE_EDGE, SQPERI,
            SRCU, VCU, SQUAE_EDGE, VSPHERE, 
            SQR_EDGE_BY_PERI, NEG_PROB]

#teranary operator
tri_op = [QAREA, VRPR, TPERI, TAREA3, SRPR]

#solve for one op
def solve(op, args):
    
    a1 = args[0]
    #unary ?
    if(op in uni_op):
        ans = ops[op](a1)
    
    #trinary operator
    elif(op in tri_op):
        a2 = args[1]
        a3 = args[2]
        ans = ops[op](a3, a2, a1)

    #binary
    else:
        a2 = args[1]
        ans = ops[op](a2, a1)
    return ans

#remove the commas
pattern = r'\b\d{1,3}(,\d{3})+\b'

# Function to remove commas from matched integers
def remove_commas(match):
    return match.group(0).replace(',', '')

#get n0, n1, .. from problem statement
def get_nums(sent):

    #remove commas
    global pattern
    sent = re.sub(pattern, remove_commas, sent)

    #match the digits
    pt = r'[-+]?\d*\.\d+|\d+'
    
    #sent = re.sub(sent, remove_commas, sent)
    matches = re.findall(pt, sent)
    numbers = [float(match) for match in matches]
    return numbers

#resolve the args
def resolve_args(argsi, ans, nums):

    #prev answers
    if('#' in argsi):
        argsi = argsi.split('#')[1].strip()
        ret = ans[int(argsi)]

    #some constant
    elif('const_' in argsi):
        argsi = argsi.strip()[6:]
        if(argsi == 'pi'):
            ret = math.pi 
        else:
            argsi = argsi.replace('_', '.')
            ret = float(argsi)

    #are they from input sentence
    elif('n' in argsi):
        w = int(argsi.strip()[1:])
        ret = nums[w]
    
    return ret 

#evaluate linear formula
def eval(x, sent):
    
    #get number from sentences
    nums = get_nums(sent)
    
    #divide in the operations
    ops = x.split('|')
    ans = []

    #iterate for every options
    for oi in ops:

        if(oi == ''):
            continue

        #split on (
        t = oi.split('(')
        
        #the operation
        op = t[0]
        
        #get the arg
        args = t[1].split(')')[0]
        args = args.split(',')
        args = [argsi.strip() for argsi in args]

        #resolve the args
        #print(op, args)
        args = [resolve_args(argsi, ans, nums) for argsi in args]

        #get the ans
        ansi = solve(op, args)
        ans.append(ansi)

    return ans[-1]

#evaluate the linear expression and add to json
def transform(data):

    right = 0
    wrong = 0
    exact_match = 0
    tr_data = []
    
    for data_i in data:
        
        #user predcition may not be executable
        try: 
            
            #copy data
            tr_data_i = data_i

            #the exact match acc
            exact_match += (tr_data_i['predicted'] == tr_data_i['linear_formula'])

            #evaluate
            tr_data_i['predicted_answer'] = eval(data_i['predicted'], data_i['Problem'])

            #print(data_i['annotated_formula'])
            err = abs((tr_data_i['predicted_answer'] - tr_data_i['answer'] )/ tr_data_i['answer'])
            
            #if beyond 2% then wrong
            if(err > 0.02 or type(tr_data_i['predicted_answer']) == complex):
                wrong += 1
                tr_data_i['predicted_answer'] = None

            #else match
            else:
                right += 1
        
            #return 
            tr_data.append(tr_data_i)
        except: 
            
            tr_data_i['predicted_answer'] = None
            wrong += 1
            tr_data.append(tr_data_i)

    print(f"Execution Accuracy: {right / (right + wrong)*100}!!" )
    print(f"Exact Match Accuracy: {exact_match / (right + wrong)*100}!!" )
    return tr_data

def main(file):

    f = open(file, 'rb')
    data = json.load(f)

    #convert the data
    tr_data = transform(data)

    #write back to file
    f = open(file, 'w')
    json.dump(tr_data, f, indent='\t', separators=(',', ': '))

if __name__ == '__main__':
    main(sys.argv[1])

'''
For the input json entry :

    {
		"Problem": "a multiple choice test consists of 4 questions , and each question has 5 answer choices . in how many r ways can the test be completed if every question is unanswered ?",
		"answer": 625,
        "linear_formula": "power(n1,n0)|"
	},

The output json file entry should contain your prediction in predicted key
and all other input fields:

    {
		"Problem": "a multiple choice test consists of 4 questions , and each question has 5 answer choices . in how many r ways can the test be completed if every question is unanswered ?",
		"answer": 625,
        "predicted": "power(n0,n0)|",
        "linear_formula": "power(n1,n0)|"
	},

We are counting on you to maintain the same files

This script will write the predicted output back to output file


    {
		"Problem": "a multiple choice test consists of 4 questions , and each question has 5 answer choices . in how many r ways can the test be completed if every question is unanswered ?",
		"answer": 625,
        "predicted": "power(n0,n0)|",
        "linear_formula": "power(n1,n0)|",
        "predicted_answer": 256
	},
'''