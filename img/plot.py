import os 
import numpy as np
import matplotlib.pyplot as plt

def matLog():
    col=['darkorchid','firebrick','limegreen']
    real_path=os.path.realpath(__file__)[:-7]
    model_file = ['model_0','model_1','model_2']
    log_file_list=sorted([log_name for log_name in os.listdir(real_path+'/'+model_file[0]) if log_name[:8]=='log_test'])

    num_log = len(log_file_list)

    episodes=[[[] for i in range(num_log)] for _ in model_file]
    rewards=[[[] for i in range(num_log)] for _ in model_file]

    for model_idx,model_path in enumerate(model_file):
        for log_idx,log_name in enumerate(log_file_list):
            with open(real_path+'/'+model_path+'/'+log_name,'rb') as f:
                for line in f:
                    par=line.split()
                    episodes[model_idx][log_idx].append(int(par[0]))
                    rewards[model_idx][log_idx].append(float(par[3]))


    rewards=np.array(rewards)
    mean_rewards=np.mean(rewards,axis=1)
    std_rewards=np.std(rewards,axis=1)
    error_bar_l=mean_rewards-std_rewards
    error_bar_h=mean_rewards+std_rewards
    print(mean_rewards)
    for b in range(len(model_file)):
        print(episodes[0][0])
        p,=plt.plot(episodes[0][0],mean_rewards[b],label=model_file[b],color=col[b])
        plt.fill_between(episodes[0][0],error_bar_l[b],error_bar_h[b],color=col[b],alpha=0.3)

    plt.xlabel('number of trainning episodes')
    plt.ylabel('average Qoe')
    plt.legend(loc='best')
    plt.savefig('mean_rewards.png')
    plt.close()

if __name__=='__main__':
    matLog()
