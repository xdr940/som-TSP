import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from path import Path
from distance import route_distance
import json
from tqdm import tqdm
def plot_network(cities, neurons, ):
    '''
    For debug
    :param cities:
    :param neurons:
    :return:
    '''
    """Plot a graphical representation of the problem"""
    mpl.rcParams['agg.path.chunksize'] = 10000

    #fig = plt.figure(figsize=(5, 5), frameon = False)
    #axis = fig.add_axes([0,0,1,1])

    #axis.set_aspect('equal', adjustable='datalim')
    #plt.axis('off')

    plt.scatter(cities['x'], cities['y'], color='blue', s=10)
    plt.scatter(cities['x'][0], cities['y'][0], color='red', s=10)

    for index, row in cities.iterrows():
        plt.text(row['x'], row['y'], int(row['city']), fontsize=10)

    plt.plot(neurons[:,0], neurons[:,1], 'r.', ls='-', color='#0063ba', markersize=2)

    plt.show()


def plot_route(input_dir):
    def back_ground():
        df = pd.read_csv(input_dir/'cities.csv')
        x = np.array(df['x'])
        y = np.array(df['y'])

        plt.figure()
        # 点标号
        for index, row in df.iterrows():
            plt.text(row['x'], row['y'], int(row['city']), fontsize=10)
        # 绘点
        plt.scatter(x, y, c='blue')
        start = df.query('city==0')
        plt.scatter(float(start['x']), float(start['y']), c='red')


    input_dir = Path(input_dir)
    files = input_dir.files('*.csv')
    files.sort()



    files = input_dir.files('*.csv')
    files.sort()
    dump_dir = input_dir / 'fig'
    dump_dir.mkdir_p()
    plt.close()
    plt.figure()
    for file in tqdm(files):
        if file.stem in['cities_nm','cities']:
            continue
        back_ground()

        df = pd.read_csv(file)

        x = df['x'].to_numpy()
        y = df['y'].to_numpy()
    # 画路径
        plt.plot(x, y, 'r')
        plt.plot([x[0], x[-1]], [y[0], y[-1]], 'r')


        plt.title(file.stem)
        plt.savefig(dump_dir/(file.stem+'.png'),bbox_inches='tight', pad_inches=0, dpi=200)
        plt.clf()
    plt.close()

def plt_traj(df,array):
    x = np.array(df['x'])
    y =np.array( df['y'])


    x_ = np.expand_dims(x,axis=1)
    y_ = np.expand_dims(y,axis=1)

    xy = np.concatenate([x_,y_],axis=1)


    #点标号
    for index,row in df.iterrows():
        plt.text(row['x'],row['y'],int(row['city']),fontsize=10)

    print(df['city'].to_numpy())
    #画路径
    plt.plot(x,y,'r')
    plt.plot([x[0],x[-1]],[y[0],y[-1]],'r')


    #绘点
    plt.scatter(x,y,c='blue')
    start = df.query('city==0')

    plt.scatter(float(start['x']),float(start['y']),c='red')

    #plt.xlim([169, 171])
    #plt.ylim([35, 37])

    plt.show()


def plot_loss(input_dir):
    input_dir = Path(input_dir)
    input_dir.mkdir_p()
    json_file = input_dir/'results.json'
    with open(json_file, encoding='utf-8') as f:
        content = f.read()
        results = json.loads(content)
    losses_decay = results['losses_decay']
    losses = results['losses']

    losses_x = list(losses.keys())
    losses_x = [float(item) for item in losses_x]

    losses_y = list(losses.values())
    losses_y = [float(item) for item in losses_y]



    losses_decay_x = list(losses_decay.keys())
    losses_decay_x = [float(item) for item in losses_decay_x]
    losses_decay_y = list(losses_decay.values())
    losses_decay_y = [float(item) for item in losses_decay_y]

    plt.close()
    plt.figure()
    plt.plot(losses_x,losses_y)
    plt.plot(losses_decay_x,losses_decay_y,'r-o')
    plt.legend(['losses','min_loss'])
    plt.title('Distance of Routes')
    plt.xlabel('Iterations')
    plt.ylabel('Distance')

    dump_dir = input_dir / 'fig'
    dump_dir.mkdir_p()

    plt.savefig(dump_dir/'losses.png',bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()

    #
    print('-----------plot_loss--------------')
    best_id = str(results['best_id'])
    best_route = results['best_route']
    min_loss = results['losses_decay'][best_id]

    print('route:')
    print([int(item) for item in best_route])
    print('distances:{}\n'.format(min_loss))


def plot_neuron_chain(input_dir):

    input_dir = Path(input_dir)

    dump_dir = input_dir / 'fig'
    dump_dir.mkdir_p()

    df = pd.read_csv(input_dir/'cities_nm.csv')


    def back_grd():




        x = np.array(df['x'])
        y = np.array(df['y'])

        x_ = np.expand_dims(x, axis=1)
        y_ = np.expand_dims(y, axis=1)

        plt.close()
        plt.figure(1)
        #绘点
        plt.scatter(x, y, c='blue')
        start = df.query('city==0')

        plt.scatter(float(start['x']), float(start['y']), c='red')
        # 点标号

        for index, row in df.iterrows():
            plt.text(row['x'], row['y'], int(row['city']), fontsize=10)


    #神经元绘制
    chain_files = input_dir.files('*.npy')
    chain_files.sort()
    axis=[]
    plt.close()
    plt.figure()
    for file in chain_files:
        chain = np.load(file)
        back_grd()
        axis.append(plt.plot(chain[:,0],chain[:,1],'b'))
        plt.title(file.stem)
        plt.savefig(dump_dir/(file.stem+'.png'), bbox_inches='tight', pad_inches=0, dpi=200)
        plt.clf()
    plt.close()


def plt_traj_p(path):
    df = pd.read_csv(path)
    x = np.array(df['x'])
    y =np.array( df['y'])


    x_ = np.expand_dims(x,axis=1)
    y_ = np.expand_dims(y,axis=1)


    #点标号
    for index,row in df.iterrows():
        plt.text(row['x'],row['y'],int(row['city']),fontsize=10)

    print(df['city'].to_numpy())
    #画路径
    plt.plot(x,y,'r')
    plt.plot([x[0],x[-1]],[y[0],y[-1]],'r')


    #绘点
    plt.scatter(x,y,c='blue')
    start = df.query('city==0')

    plt.scatter(float(start['x']),float(start['y']),c='red')

    #plt.xlim([169, 171])
    #plt.ylim([35, 37])
    plt.show()

    pass



def plt_traj_np(args):
    p = Path(args.data_dir)/'data1.csv'
    df = pd.read_csv(p)
    route = np.array(args.route_plt)
    df =df.reindex(route)
    dis = route_distance(df)
    print("--> route distance:{}".format(dis))
    x = np.array(df['x'])
    y =np.array( df['y'])




    #点标号
    for index,row in df.iterrows():
        plt.text(row['x'],row['y'],int(row['city']),fontsize=10)

    print(df['city'].to_numpy())
    #画路径
    plt.plot(x,y,'r')
    plt.plot([x[0],x[-1]],[y[0],y[-1]],'r')


    #绘点
    plt.scatter(x,y,c='blue')
    start = df.query('city==0')
    plt.title('Best Route')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.scatter(float(start['x']),float(start['y']),c='red')

    #plt.xlim([169, 171])
    #plt.ylim([35, 37])

    plt.show()

    pass


def plt_mtsp(args):
    p = Path(args.data_dir) / 'data1.csv'
    df = pd.read_csv(p)

    # 点标号
    for index, row in df.iterrows():
        plt.text(row['x'], row['y'], int(row['city']), fontsize=10)
    print(df['city'].to_numpy())

    x=df['x']
    y=df['y']
    # 绘点
    plt.scatter(x, y, c='blue')
    start = df.query('city==0')
    plt.title('Best Routes')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.scatter(float(start['x']), float(start['y']), c='red')

    routes = args.routes

    routes_nps = []
    for route in routes:
        route = np.array(route)
        routes_nps.append(route)
    dfs=[]
    dises=[]
    xs=[]
    ys=[]
    colors=['r','g','b','k']
    for idx,item in enumerate(routes_nps):
        df_ = df.reindex(item)
        dfs.append(df_)
        dises.append(route_distance(df_))
        print("--> route distance:{}".format(dises[idx]))
        xs.append(np.array(dfs[-1]['x']))
        ys.append(np.array(dfs[-1]['y']))
        plt.plot(xs[-1],ys[-1],colors[idx])
        plt.plot([xs[-1][0],xs[-1][-1]],[ys[-1][0],ys[-1][-1]],colors[idx])

    print(np.array(dises).sum())





    # plt.xlim([169, 171])
    # plt.ylim([35, 37])

    plt.show()
