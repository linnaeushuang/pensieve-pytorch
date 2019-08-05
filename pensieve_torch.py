import os
import logging
import argparse
import numpy as np
import multiprocessing as mp
import env
import load_trace
import torch
from A3C import A3C
from datetime import datetime
import time


S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
NUM_AGENTS = 2
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
HD_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
SUMMARY_DIR = './results'
LOG_FILE = './results/log'
TEST_LOG_FOLDER = './test_results/'
TRAIN_TRACES = './data/cooked_traces/'

#CRITIC_MODEL= './results/critic.pt'
#ACTOR_MODEL = './results/actor.pt'
CRITIC_MODEL = None

TOTALEPOCH=300
IS_CENTRAL=True
NO_CENTRAL=False

def testing(epoch, actor_model,log_file):
    # clean up the test results folder
    os.system('rm -r ' + TEST_LOG_FOLDER)
    os.system('mkdir ' + TEST_LOG_FOLDER)
    
    # run test script
    os.system('python rl_test.py '+actor_model)
    
    # append test performance to the log
    rewards = []
    test_log_files = os.listdir(TEST_LOG_FOLDER)
    for test_log_file in test_log_files:
        reward = []
        with open(TEST_LOG_FOLDER + test_log_file, 'r') as f:
            for line in f:
                parse = line.split()
                try:
                    reward.append(float(parse[-1]))
                except IndexError:
                    break
        rewards.append(np.sum(reward[1:]))

    rewards = np.array(rewards)

    rewards_min = np.min(rewards)
    rewards_5per = np.percentile(rewards, 5)
    rewards_mean = np.mean(rewards)
    rewards_median = np.percentile(rewards, 50)
    rewards_95per = np.percentile(rewards, 95)
    rewards_max = np.max(rewards)

    log_file.write(str(epoch) + '\t' +
                   str(rewards_min) + '\t' +
                   str(rewards_5per) + '\t' +
                   str(rewards_mean) + '\t' +
                   str(rewards_median) + '\t' +
                   str(rewards_95per) + '\t' +
                   str(rewards_max) + '\n')
    log_file.flush()


def central_agent(net_params_queues, exp_queues, model_type):
    torch.set_num_threads(1)

    timenow=datetime.now()
    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS

    logging.basicConfig(filename=LOG_FILE + '_central',
                        filemode='w',
                        level=logging.INFO)


    net=A3C(IS_CENTRAL,model_type,[S_INFO,S_LEN],A_DIM,ACTOR_LR_RATE,CRITIC_LR_RATE)
    test_log_file=open(LOG_FILE+'_test','a')

    if CRITIC_MODEL is not None and os.path.exists(ACTOR_MODEL):
        net.actorNetwork.load_state_dict(torch.load(ACTOR_MODEL))
        net.criticNetwork.load_state_dict(torch.load(CRITIC_MODEL))


    epoch=0
    while epoch<TOTALEPOCH:
        # synchronize the network parameters of work agent
        actor_net_params=net.getActorParam()
        #critic_net_params=net.getCriticParam()
        for i in range(NUM_AGENTS):
            #net_params_queues[i].put([actor_net_params,critic_net_params])
            net_params_queues[i].put(actor_net_params)
            # Note: this is synchronous version of the parallel training,
            # which is easier to understand and probe. The framework can be
            # fairly easily modified to support asynchronous training.
            # Some practices of asynchronous training (lock-free SGD at
            # its core) are nicely explained in the following two papers:
            # https://arxiv.org/abs/1602.01783
            # https://arxiv.org/abs/1106.5730

        # record average reward and td loss change
        # in the experiences from the agents
        total_batch_len = 0.0
        total_reward = 0.0
        total_td_loss = 0.0
        total_entropy = 0.0
        total_agents = 0.0 

        # assemble experiences from the agents
        actor_gradient_batch = []
        critic_gradient_batch = []

        for i in range(NUM_AGENTS):
            s_batch, a_batch, r_batch, terminal, info = exp_queues[i].get()


            net.getNetworkGradient(np.stack(s_batch[1:],axis=0),np.vstack(a_batch[1:]),np.vstack(r_batch[1:]),terminal=terminal)


            total_reward += np.sum(r_batch)
            total_batch_len += len(r_batch)
            total_agents += 1.0
            total_entropy += np.sum(info['entropy'])

        # log training information
        net.updateNetwork()

        epoch += 1
        avg_reward = total_reward  / total_agents
        avg_entropy = total_entropy / total_batch_len

        logging.info('Epoch: ' + str(epoch) +
                     ' Avg_reward: ' + str(avg_reward) +
                     ' Avg_entropy: ' + str(avg_entropy))

        if epoch % MODEL_SAVE_INTERVAL == 0:
            # Save the neural net parameters to disk.
            print("\nTrain ep:"+str(epoch)+",time use :"+str((datetime.now()-timenow).seconds)+"s\n")
            timenow=datetime.now()
            torch.save(net.actorNetwork.state_dict(),SUMMARY_DIR+"/actor.pt")
            if model_type<2:
                torch.save(net.criticNetwork.state_dict(),SUMMARY_DIR+"/critic.pt")
            testing(epoch,SUMMARY_DIR+"/actor.pt",test_log_file)

        #if epoch % 200 ==0 :
        #    time.sleep(25)


def agent(agent_id, all_cooked_time, all_cooked_bw, net_params_queue, exp_queue, model_type):
    torch.set_num_threads(1)

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw,
                              random_seed=agent_id)

    with open(LOG_FILE+'_agent_'+str(agent_id),'w') as log_file:


        net=A3C(NO_CENTRAL,model_type,[S_INFO,S_LEN],A_DIM,ACTOR_LR_RATE,CRITIC_LR_RATE)

        # initial synchronization of the network parameters from the coordinator
        actor_net_params= net_params_queue.get()
        net.hardUpdateActorNetwork(actor_net_params)

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY

        action_vec = np.zeros(A_DIM)
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []

        time_stamp = 0

        epoch=0
        while True:  # experience video streaming forever

            # the action is from the last decision
            # this is to make the framework similar to the real
            delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = \
                net_env.get_video_chunk(bit_rate)

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            # -- linear reward --
            # reward is video quality - rebuffer penalty - smoothness
            reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                     - REBUF_PENALTY * rebuf \
                     - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                               VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

            # -- log scale reward --
            # log_bit_rate = np.log(VIDEO_BIT_RATE[bit_rate] / float(VIDEO_BIT_RATE[-1]))
            # log_last_bit_rate = np.log(VIDEO_BIT_RATE[last_bit_rate] / float(VIDEO_BIT_RATE[-1]))

            # reward = log_bit_rate \
            #          - REBUF_PENALTY * rebuf \
            #          - SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate)

            # -- HD reward --
            # reward = HD_REWARD[bit_rate] \
            #          - REBUF_PENALTY * rebuf \
            #          - SMOOTH_PENALTY * np.abs(HD_REWARD[bit_rate] - HD_REWARD[last_bit_rate])

            r_batch.append(reward)

            last_bit_rate = bit_rate

            # retrieve previous state
            if len(s_batch) == 0:
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            # this should be S_INFO number of terms
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
            state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
            state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

            # compute action probability vector
            action_prob = net.actionSelect(np.reshape(state, (1, S_INFO, S_LEN)))
            action_cumsum = np.cumsum(action_prob)
            bit_rate = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
            # Note: we need to discretize the probability into 1/RAND_RANGE steps,
            # because there is an intrinsic discrepancy in passing single state and batch states

            entropy_record.append(3)

            # log time_stamp, bit_rate, buffer_size, reward
            log_file.write(str(time_stamp) + '\t' +
                           str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                           str(buffer_size) + '\t' +
                           str(rebuf) + '\t' +
                           str(video_chunk_size) + '\t' +
                           str(delay) + '\t' +
                           str(reward) + '\n')
            log_file.flush()

            # report experience to the coordinator
            if len(r_batch) >= TRAIN_SEQ_LEN or end_of_video:

                exp_queue.put([s_batch[1:],  # ignore the first chuck
                               a_batch[1:],  # since we don't have the
                               r_batch[1:],  # control over it
                               end_of_video,
                               {'entropy': entropy_record}])
                
                epoch+=1
                if epoch<TOTALEPOCH:
                    actor_net_params=net_params_queue.get()
                    net.hardUpdateActorNetwork(actor_net_params)
                    del s_batch[:]
                    del a_batch[:]
                    del r_batch[:]
                    del entropy_record[:]
                    log_file.write('\n')  # so that in the log we know where video ends
                else:
                    break

            # store the state and action into batches
            if end_of_video:
                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here

                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1

                s_batch.append(np.zeros((S_INFO, S_LEN)))
                a_batch.append(action_vec)

            else:
                s_batch.append(state)

                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1
                a_batch.append(action_vec)


def main(arglist):

    time=datetime.now()
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)


    assert len(VIDEO_BIT_RATE) == A_DIM

    # create result directory
    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    # inter-process communication queues
    net_params_queues = []
    exp_queues = []
    for i in range(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues,arglist.model_type))
    coordinator.start()

    all_cooked_time, all_cooked_bw, _ = load_trace.load_trace(TRAIN_TRACES)
    agents = []
    for i in range(NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(i, all_cooked_time, all_cooked_bw,
                                       net_params_queues[i],
                                       exp_queues[i],arglist.model_type)))
    for i in range(NUM_AGENTS):
        agents[i].start()

    # wait unit training is done
    coordinator.join()
    for i in range(NUM_AGENTS):
        agents[i].join()

    print(str(datetime.now()-time))

def parse_args():
    parser=argparse.ArgumentParser("Pensieve")
    parser.add_argument("--model_type",type=int,default=0,help="Refer to README for the meaning of this parameter")
    return parser.parse_args()

if __name__ == '__main__':
    arglist=parse_args()
    main(arglist)
