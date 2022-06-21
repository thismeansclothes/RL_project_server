import time
import csv
import argparse
import torch

import numpy as np
from sac_agent import SACAgent
from utils import set_log_dir
import random

from graph.graph_lib import Graph_Lib

def run_sac(
            max_iter=1e6,
            eval_interval=2000,
            start_train=10000,
            train_interval=50,
            buffer_size=1e6,
            fill_buffer=20000,
            truncate=1000,
            gamma=0.99,
            pi_lr=3e-4,
            q_lr=3e-4,
            polyak=5e-3,
            alpha=0.2,
            hidden1=256,
            hidden2=256,
            batch_size=128,
            device='cpu',
            render='False'
            ):

    params = locals()

    max_iter = int(max_iter)
    buffer_size = int(buffer_size)

    env = Graph_Lib()
    env_id = "GC_SAC"

    dimS = 24
    dimA = 1
    ctrl_range = 500
    max_ep_len = 120

    # dimS, dimA, ctrl_range, max_ep_len = get_env_spec(env)

    if truncate is not None:
        max_ep_len = truncate

    agent = SACAgent(
                     dimS,
                     dimA,
                     ctrl_range,
                     gamma=gamma,
                     pi_lr=pi_lr,
                     q_lr=q_lr,
                     polyak=polyak,
                     alpha=alpha,
                     hidden1=hidden1,
                     hidden2=hidden2,
                     buffer_size=buffer_size,
                     batch_size=batch_size,
                     device=device,
                     render=render
                     )

    set_log_dir(env_id)

    num_checkpoints = 5
    checkpoint_interval = max_iter // (num_checkpoints - 1)
    current_time = time.strftime("%m%d-%H%M%S")
    train_log = open('./train_log/' + env_id + '/SAC_' + current_time + '.csv',
                     'w', encoding='utf-8', newline='')

    path = './eval_log/' + env_id + '/SAC_' + current_time
    eval_log = open(path + '.csv', 'w', encoding='utf-8', newline='')

    train_logger = csv.writer(train_log)
    eval_logger = csv.writer(eval_log)

    with open(path + '.txt', 'w') as f:
        for key, val in params.items():
            print(key, '=', val, file=f)

    obs = env
    step_count = 0
    ep_reward = 0

    min_nodes = 100
    max_nodes = 500

    node_cnts = obs.insert_batch(min_nodes, max_nodes) # 그래프 임의로 생성

    obs.init_node_embeddings()
    obs.init_graph_embeddings()

    colored_arrs =[False]*node_cnts

    max_colors = -1
    done = False

    # main loop
    start = time.time()
    for t in range(max_iter + 1):

        next_obs = obs
        node_embed = obs.get_node_embed()
        graph_embed = obs.get_graph_embed()


        if(t >= node_cnts):
            done = True
            action = -1
            continue

        if t < fill_buffer:
            found = False
            while not found:
                action = random.randint(0, node_cnts -1)
                if not colored_arrs[action]:
                    found = True
                    colored_arrs[action] = True
                    embeddings = np.concatenate([node_embed[node], graph_embeds])
                    embeddings = torch.from_numpy(embeddings).float()

        else:
            max_action = -9999
            max_node = -1
            node_np = np.array(node_embeds[:,6])
            node_np = node_np.argsort()[-10:][::-1]

            for node in node_np:
                if colored_arrs[node]:
                    continue
                embeddings = np.concatenate([node_embeds[node], graph_embeds])
                embeddings = torch.from_numpy(embeddings).float()
                action = agent.act(embeddings)

                if(max_action < action):
                    max_node = node
                    max_action = action

            colored_arrs[max_node] = True
            action = max_node

        colors = next_obs.color_batch(action)
        rewards = 0

        if(colors == -1):
            rewards = -9999
        else:
            rewards = - max(0, - max_colors + colors)
            if(max_colors < colors):
                max_colors = colors


        # next_obs, reward, done, _ = env.step(action)
        step_count += 1

        if step_count == max_ep_len:
            done = False

        agent.buffer.append(obs, action, next_obs, rewards, done)

        obs = next_obs
        ep_reward += rewards

        if done or (step_count == max_ep_len):
            train_logger.writerow([t, ep_reward])
            env.reset_batch()
            obs = env
            step_count = 0
            ep_reward = 0

        if (t >= start_train) and (t % train_interval == 0):
            for _ in range(train_interval):
                agent.train()
                print('in train...')

        if t % eval_interval == 0:
            eval_score = eval_agent(agent)
            log = [t, eval_score]
            print('step {} : {:.4f}'.format(t, eval_score))
            eval_logger.writerow(log)

        if t % checkpoint_interval == 0:
            agent.save_model('./checkpoints/' + env_id + '/sac_{}th_iter_'.format(t))

    train_log.close()
    eval_log.close()

    return


def eval_agent(agent, eval_num=5):
    log = []
    for ep in range(eval_num):
        env = Graph_Lib()

        node_cnts = env.insert_batch(min_nodes, max_nodes)

        env.init_node_embeddings()
        env.init_graph_embeddings()

        colored_arrs = [False]*node_cnts
        max_node =-1

        max_colors = -1
        reward = 0

        state = env
        next_state = state

        step_count = 0
        ep_reward = 0
        done = False

        while not done:
            if(step_count >= node_cnts):
                done = True
                action = -1
                continue

            max_action = -9999
            max_node = -1
            node_np = np.array(node_embeds[:,6])
            node_np = node_np.argsort()[-10:][::-1]

            for node in node_np:
                if colored_arrs[node]:
                    continue
                embeddings = np.concatenate([node_embeds[node], graph_embeds])
                embeddings = torch.from_numpy(embeddings).float()
                action = agent.act(embeddings,eval=True)

                if(max_action < action):
                    max_node = node
                    max_action = action

            colored_arrs[max_node] = True
            action = max_node
            
            colors = next_state.color_batch(action)
            rewards = 0

            if(colors == -1):
                rewards = -9999
            else:
                rewards = - max(0, - max_colors + colors)
                if(max_colors < colors):
                    max_colors = colors

            step_count += 1
            state = next_state
            ep_reward += reward

        log.append(ep_reward)

    avg = sum(log) / eval_num

    return avg



if __name__ == '__main__':

    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser()

    parser.add_argument('--truncate', required=False, default=1000, type=int)
    parser.add_argument('--device', required=False, default=default_device)
    parser.add_argument('--max_iter', required=False, default=5e5, type=float)
    parser.add_argument('--eval_interval', required=False, default=2000, type=int)
    parser.add_argument('--render', required=False, default=False, type=bool)
    parser.add_argument('--tau', required=False, default=5e-3, type=float)
    parser.add_argument('--lr', required=False, default=3e-4, type=float)
    parser.add_argument('--hidden1', required=False, default=256, type=int)
    parser.add_argument('--hidden2', required=False, default=256, type=int)
    parser.add_argument('--train_interval', required=False, default=50, type=int)
    parser.add_argument('--start_train', required=False, default=10000, type=int)
    parser.add_argument('--fill_buffer', required=False, default=20000, type=int)

    args = parser.parse_args()

    run_sac(
            max_iter=args.max_iter,
            eval_interval=args.eval_interval,
            start_train=args.start_train,
            train_interval=args.train_interval,
            fill_buffer=args.fill_buffer,
            truncate=args.truncate,
            gamma=0.99,
            pi_lr=args.lr,
            q_lr=args.lr,
            polyak=args.tau,
            alpha=0.2,
            hidden1=args.hidden1,
            hidden2=args.hidden2,
            batch_size=128,
            buffer_size=1e6,
            device=args.device,
            render=args.render
            )
