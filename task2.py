import numpy as np
from typing import List, Optional, Dict, Tuple
import math

# =========================================================
# ===============   ENVIRONMENT (Poisson)   ===============
# =========================================================

class PoissonDoorsEnv:
    """
    This creates a Poisson environment. There are K doors and each has an associated mean.
    In each step you pick an arm i. Damage to a door is drawn from its corresponding
    Poisson Distribution. Initial health of each door is H0 and decreases by damage in each step.
    Game ends when any door's health < 0.
    """
    def __init__(self, mus: List[float], H0: int = 100, rng: Optional[np.random.Generator] = None):
        self.mus = np.array(mus, dtype=float)
        assert np.all(self.mus > 0), "Poisson means must be > 0"
        self.K = len(mus)
        self.H0 = H0
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset()

    def reset(self):
        self.health = np.full(self.K, self.H0, dtype=float)
        self.t = 0
        return self.health.copy()

    def step(self, arm: int) -> Tuple[float, bool, Dict]:
        reward = float(self.rng.poisson(self.mus[arm]))
        self.health[arm] -= reward
        self.t += 1
        done = np.any(self.health < 0.0)
        return reward, done, {"reward": reward, "health": self.health.copy(), "t": self.t}


# =========================================================
# =====================   POLICIES   ======================
# =========================================================

class Policy:
    """
    Base Policy interface.
    - Implement select_arm(self, t) to return an int in [0, K-1] to choose an arm.
    - Optionally override update(...) for custom learning.
    """
    def __init__(self, K: int, rng: Optional[np.random.Generator] = None):
        self.K = K
        self.rng = rng if rng is not None else np.random.default_rng()
        self.counts = np.zeros(K, dtype=int)
        self.sums   = np.zeros(K, dtype=float)

    def reset_stats(self):
        self.counts[:] = 0
        self.sums[:]   = 0.0

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        self.sums[arm]   += reward

    @property
    def means(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.sums / np.maximum(self.counts, 1)

    def select_arm(self, t: int) -> int:
        raise NotImplementedError
    
# GLOBAL FUNCTIONS FOR KL-UCB

def kl_divergence_poission(p,q):
    eps=1e-10
    p = np.clip(p, a_max=1-eps, a_min = eps)
    q = np.clip(q, a_max = 1-eps, a_min = eps)
    return q-p + p*math.log(p/q)
    # return p*math.log(p/q)+(1-p)*math.log((1-p)/(1-q))

def find_q(arm_index, t, ut,p, c, eps,max_it):

    val = (math.log(t)+c*math.log(max(math.log(max(t,2)),1.0000001)))/ut
    low = p
    high = 3.0
    for _ in range (max_it) :
        mid = 0.5*(high+low)
        if kl_divergence_poission(p,mid)<=val:
            low = mid
        else:
            high = mid
        
    return 0.5*(high+low)

## TASK 2: Make changes here to implement your policy ###
class StudentPolicy(Policy):
    """
    Implement your own algorithm here.
    Replace select_arm with your strategy.
    Currently it has a simple implementation of the epsilon greedy strategy.
    Change this to implement your algorithm for the problem.
    """
    def __init__(self, K: int, rng: Optional[np.random.Generator] = None):
        super().__init__(K, rng)
        # self.eps = 0.1
        # self.c=0.25
    
    # def hellinger_sq(self, mu_hat,mu):
    #     return 1 - np.exp(-0.5*(mu_hat+mu)+np.sqrt(mu_hat*mu))
    
    # def find_ucb(self, mu_hat, count , t, max_iters):
       
    #     target = 1-np.exp(-self.c*np.log(t)/count)
    #     lower = mu_hat
    #     upper = 3.0

    #     if self.hellinger_sq(mu_hat, upper)<=target:
    #         return upper  
        
    #     for _ in range(max_iters):
    #         mid = 0.5*(lower + upper)
    #         if self.hellinger_sq(mu_hat,mid)<=target:
    #             lower=mid
    #         else:
    #             upper = mid
    #     return 0.5*(lower+upper)



    def select_arm(self, t: int) -> int:

        #UCB
        # if np.sum(self.counts)<self.K:
        #     arm_indx = int(np.sum(self.counts))
        #     return arm_indx
        # else:
        #     t=np.sum(self.counts)
        #     ucb = self.means + np.sqrt(2*np.log(t)/(self.counts+1e-7))
        #     return np.argmax(ucb)
        #THOMPSON
        # beta = self.rng.beta(self.success+1,self.failures+1)              
        # return np.argmax(beta)
        
        #HALLINGER_UCB
        # index = np.zeros(self.K)
        # door_strengths=[100-self.sums[i] for i in range(self.K)]
        # if np.sum(self.counts)<self.K:
        #     arm_indx = int(np.sum(self.counts))
        #     return arm_indx
        # else:
        #     mu_hat = self.means
        #     optimal_arm=0
        #     optimal_mu=0
        #     for i in range(self.K):
        #         hallinger_mu = self.find_ucb(self.means[i],int(self.counts[i]), int(np.sum(self.counts)),10)
        #         index[i] = hallinger_mu/max(door_strengths[i], 1e-6)
        #     return int(np.argmax(index)) 

        # KL-UCB(poisson)
        index = np.zeros(self.K)
        door_strengths=[100-self.sums[i] for i in range(self.K)]
        if np.sum(self.counts)<self.K:
            arm_indx = int(np.sum(self.counts))
            return arm_indx
        else:
            
            for i in range (self.K):
                q = find_q(i, t= int(np.sum(self.counts)), ut= self.counts[i], p = self.means[i], c=3.5, eps=1e-6, max_it =5)
                index[i] = q/max(door_strengths[i], 1e-6)
            return int(np.argmax(index)) 

    def update(self, arm: int, reward: float):
        super().update(arm, reward)