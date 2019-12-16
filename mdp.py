import numpy as np
import itertools as it

def value_iteration(r, g, P, target_error=0, num_iters=None):
    u = np.zeros(r.shape)
    error = np.inf

    for i in it.count():

        if i == num_iters: break
        if error < target_error: break

        u_new = r + g * (P * u[np.newaxis,np.newaxis,:]).sum(axis=2).max(axis=0)
        error = np.fabs(u_new - u).max() * (1 - g) / g
        u = u_new
        
        print("iter %d error = %f, u_new= %s"%(i,error,u))
        # print(u)
    
    return u

def policy_iteration(r, g, P, num_iters=None):

    pi = np.zeros(r.shape).astype(int)
    # pi = np.random.choice(P.shape[0], size=r.shape).flatten()

    for i in it.count():

        if i == num_iters: break

        # policy evaluation
        M = np.empty(P.shape[1:])
        for s in range(len(pi)):
            M[s,:] = g * P[pi[s],s,:]
        M = np.eye(*M.shape) - M
        u = np.linalg.solve(M, r)
    
        print("iter %d pi=%s, u= %s"%(i,pi,u))

        # policy improvement (one-step look-ahead)
        pi_new = (P*u[np.newaxis,np.newaxis,:]).sum(axis=2).argmax(axis=0)
        if (pi == pi_new).all(): break
        pi = pi_new
    
    return pi, u

if __name__ == "__main__":
    
    
    # 2-state exercise

    r = np.array([-1,1]) # s_A, s_B
    g = .5 # gamma
    # P[a,i,j] = Pr(St+1 = sj | St=si, At=a)
    P = np.array([
        # [[.75, .25], # a1: stay
        #  [.25, .75]],

        [[.8, .2], # a1: stay
         [.4, .6]],

        # [[.25, .75], # a2: switch
        #  [.75, .25]]])

        [[.2, .8], # a2: switch
         [.75, .25]]])

    print('r,P:')
    print(r)
    print(P)

    print("value iteration:")
    u = value_iteration(r,g,P,num_iters=10)
    
    print("policy iteration:")
    pi, u = policy_iteration(r,g,P,num_iters=10)

