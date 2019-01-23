import numpy as np
import random
from numpy.linalg import inv

def transition(rain_t):
    return 0.7 if rain_t else 0.3

def observation(rain_t):
    return 0.9 if rain_t else 0.2

def direct_sampling(seq_length):
    ''' starting from the initial probability of an event 
        samples along the sequence '''
    # first element of the raw = state
    # second element of the raw = observation
    seq_matrix = np.zeros((seq_length, 2))

    # first state
    outcome1 = True if random.random() <= 0.5 else False

    for i in range(seq_length):
        outcome1 = True if random.random() <= transition(outcome1) else False
        seq_matrix[i,0] = outcome1
        outcome2 = True if random.random() <= observation(outcome1) else False
        seq_matrix[i,1] = outcome2
    return seq_matrix

def forward(f, obs):
    # transition  matrix
    T = np.matrix([[0.7, 0.3],[0.3, 0.7]])
    # observation matrix
    O = [[0.9,  0],[0,  0.2]] if obs else [[0.1, 0], [0, 0.8]]
    forw = O*T.transpose()*f
    forw /= np.sum(forw)
    return forw

def backward(b, obs):
    # transition  matrix
    T = np.matrix([[0.7, 0.3],[0.3, 0.7]])
    # observation matrix
    O = [[0.9,  0],[0,  0.2]] if obs else [[0.1, 0], [0, 0.8]]
    back = T*O*b
    back /= np.sum(back)
    return back

def forw_backw(state_sequence, k):
    ''' - builds the forward and the backward messages
        - multiplies elementwise the messages '''

    f = np.matrix([0.5, 0.5]).transpose()
    forw_mess = f.transpose()
    length = len(state_sequence)
    for i in range(length):
        f = forward(f, state_sequence[i,1])
        forw_mess = np.concatenate((forw_mess,f.transpose()))

    b = np.matrix([1.0, 1.0]).transpose()
    #back_mess = b.transpose()
    back_mess = np.zeros((length, 2))
    back_mess[-1] = b.transpose()
    for i in range(length-2, k-1, -1):
        b = backward(b, state_sequence[i-1,1])
        back_mess[i] = b.transpose()
    #print(back_mess)

    sequence = np.zeros((length,2))
    for i in range(length):
        #print(forw_mess[i])
        fxb = np.multiply(forw_mess[i], back_mess[i])
        fxb /= np.sum(fxb)
        sequence[i] = fxb
    return sequence

def forward_improved(f, obs):
    ''' for the improved version of the forw-backw algo.
        Computes the forward messages "backward"
    '''
    # transition  matrix
    T = np.matrix([[0.7, 0.3],[0.3, 0.7]])
    # observation matrix
    O = [[0.9,  0],[0,  0.2]] if obs else [[0.1, 0], [0, 0.8]]
    forw = inv(T.transpose())*inv(O)*f
    forw = inv(O*T.transpose())*f
    forw /= np.sum(forw)
    #print('observation:', obs, 'prediction: ', forw)
    return forw

def forw_backw_improved(state_sequence, k):
    ''' Used to save memory.
        Computes the LAST forward messages 
        (it does not store the whole sequence of messages)
        The computes from the end the elementwise multiplication "forw x backw"
        PROBLEM : inverse matrix produces negative probabilities. 
    '''
    f = np.matrix([0.5, 0.5]).transpose()

    b = np.matrix([1.0, 1.0]).transpose()
    length = len(state_sequence)

    # compute the last forward message
    for i in range(length):
        f = forward(f, state_sequence[i,1])

    sequence = np.zeros((length,2))
    fxb = np.multiply(f.transpose(), b.transpose())
    # fxb /= sum(fxb)
    sequence[-1] = fxb
    for i in range(length-2, k-1, -1):
        b = backward(b, state_sequence[i-1,1])
        f = forward_improved(f, state_sequence[i-1,1])
        fxb = np.multiply(f.transpose(), b.transpose())
        # fxb /= sum(fxb)
        sequence[i] = fxb
    return sequence

def main():
    seq_length = 20
    num_sequences = 15
    # Direct Sampling
    state_sequences =[]
    for _ in range(num_sequences):
        state_sequences.append(direct_sampling(seq_length))
    # Takes a sequence obtained via direct sampling -> forward-backward algo
    result = forw_backw(state_sequences[1], 0)
    print('Normal algo')
    print(result)
    # Takes a sequence obtained via direct sampling -> improved forward-backward algo (it does not work :( )
    result_impr = forw_backw_improved(state_sequences[1], 0)
    print('Improved algo')
    print(result_impr)

if __name__ == '__main__':
    main()