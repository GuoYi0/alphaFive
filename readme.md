# AlphaFive
模仿AlphaGo/AlphaGo Zero写的一个五子棋AI，为了快速收敛，针对五子棋的特点做了一些小trick改进

先上效果图
![five_6960.gif](https://ra)

## 运行
*训练* `python train.py`
*自我对弈* `python self_play.py`
*人机对弈* `python GUI.py`
*两个ckpt之间对弈或者在所有ckpt之间选择一个最佳* `python choose_best_player.py`

## 算法
AlphaGo/AlphaGo Zero核心思想是用一个network搭配MCST进行搜索和优化。network的输入为相对于当前玩家棋盘局面，
输出为在各个地方落子的概率（即policy）和当前局面对于当前玩家而言最终的得分期望（即value）,若最终当前玩家输了，得-1分，
赢了得+1分，和棋得0分。故value是介于-1到1之间的一个实数。

policy的作用是为MCTS提供一个先验概率，让MCTS优先搜索对当前选手而言更可能获胜的路径，也就是说基于当前策略去采样，而不是随机采样；value的作用在于搜索到叶子节点的时候，若没有game over，则以value值进行回溯更新。
单纯的MCTS在搜索到非game over的叶子节点的时候会进行roll out，进行一次路径采样，用这个采样结果来估计基于当前局面当前选手的得分期望。
虽然这个估计是无偏估计，但是仅仅用一个样本来估计期望显然具有很大的方差。而value值虽然可能是有偏的，但是他是依据很多次棋局训练出来的，具有一定的平均意义，方差相对较小。

以某种棋局状态出发进行多次MCTS搜索以后，就可以依据各个子节点的访问次数构造一个概率分布，依据该概率分布来决定真正应该再何处落子，同时该概率分布也可以作为network训练policy的监督信号。
当一局棋结束以后，就可以知道该轮对弈在每个棋局状态下的得分，该得分将作为训练value的监督信号。

从以上算法可知，这是一个不断根据network的输出以MCST进行对弈采样，然后把对弈结果再拿来更新network的参数，这样一个不断迭代过程。

## 特点
（该project的运行环境是GTX 1070显卡和六核i7-8700 CPU的笔记本电脑。对于强化学习来说穷得不该入这行的大门。用主进程训练network，五个子进程生成模拟数据，一条长度为30的对弈数据大约就得30秒钟。故有些特点是我自己加进去的，感觉可能会加快运行速度和收敛）

[1] *输入特征*  围棋的落子规则决定了当前落子不仅仅依赖于当前局面，也依赖于过往局面，故AlphaGo/AlphaGo Zero必须要把过往的棋局和当前棋局叠在一起作为network的输入（当然，叠加更多的过往棋局也可能利于训练）。
此外，围棋的先手和后手的最终判输赢的规则也不一样，故也需要告诉network当前玩家是先手还是后手。对于五子棋而言，落子仅仅取决于当前棋局，与过往棋局完全无关，此外先后手最终的判输赢规则是一样的，所以仅仅需要输入当前局面即可。
对于11*11的棋盘，输入shape可以是[B, 2, 11, 11]，其中B是bachsize，2个channel其中一个是当前玩家的特征，另一个是对方玩家的特征，有棋子的地方为1，没有棋子的地方为0。
理论上只需要一个channel就可以的，当前玩家的棋子为1，对方玩家的棋子为-1，没有棋子的地方是0。搞两个channel可以加速收敛。此外，该project加了第三个channel，其在对方玩家最后一次落子的地方为1，其余地方为0，即表征last action。这个channel可以起到一个attention的作用，告诉当前玩家可能需要聚焦于对方玩家的落子点的附近进行落子。
这个channel可能没啥太大的卵用，后续可以做对比实验试一试。

[2] **MCST树设计**  一般来说蒙特卡洛搜索树是一个树状结构，但由于五子棋的落子决策完全仅依赖于当前状态（有last action的情形除外），而不同落子顺序可能到达相同的状态，这个相同的状态的状态信息就可以复用了。
故本project并没有设计成树状结构，而是以dict的形式存储，其中key为一个字符串表示的某种状态，value是该状态的状态信息。从不同路径抵达该状态时可以共享该信息，并共同更新该信息。但是在有last action的时候，情况有些微妙的变化。
last action仅仅在需要作为network的输入的时候起作用。在模拟对弈的时候，到达某一个局面以后，假设需要以当前局面为根节点出发进行500次搜索，这500次的last action是一样的，搜索完毕以后在该节点形成的概率分布将作为policy的监督信号，该监督信号都对应于同一个last action，这一点是没有问题的。但是当该局面节点曾经作为叶子节点的时候，对叶子节点的评估所使用的last action就未必是现在的last action了。
只有当对弈局数足够多以后，这个影响才可以逐渐减弱。后续可以去掉last action这个channel一试。

[3] **数据处理** 五子棋有一个很大的bug，就是（貌似35步以内）先手必赢。这样产生的后果是，模拟对弈的数据里面，先手赢的数据量会多于后手赢的数据量，这样失衡的数据直接拿去训练，会导致网络进一步偏好先手赢（如果当前玩家落子前，棋局里面当前玩家的棋子数量等于对手玩家的棋子数量，则当前玩家就是先手；若当前玩家的棋子数量比对手棋子数量少一个，则是后手。故网络完全可以通过棋子数量学到当前玩家是先手还是后手。），这种偏好进一步让模拟对弈产生更多的先手赢的棋局。
最终模型可能会坍塌，即先手预测的value接近于1，policy是比较准确；后手预测的value接近于-1，policy的概率分布的熵较大。坍塌以后，会产生大量长度只有9或者11的棋局。当然，如果噪声足够大，训练时间足够长，这种现象可以缓解。本project采取一个的缓解方案是，记录replay buffer里面先手和后手赢棋的棋局数，当某一方赢棋数量太少的时候，若搜集到该方的一条赢棋，则重复加入buffer。此外，对于步数太短的棋局，以一定概率舍弃。这一部分在`utils.py RandomStack`里面。

[4] **训练权重** 在不断模拟对弈过程中，越是往后的的局面出现的频次就会越小，越是靠前的局面出现的频次肯定越大。故在本project设计了一个权重增长因子，使得靠后的局面获得的训练权重大于靠前的局面的训练权重。这样做的另一个原因是，靠后的落子与输赢的关联性可能更大，所以获得一个较大的训练权重。

[5] **探索方案** 在训练过程中，很容易使得网络输出的policy的熵太小，不利于充分探索。缓解方案有很多，例如把熵加入loss里面，加大噪声等等。本project比较暴力，在根节点处强制要求每个子节点至少被访问两次。这样一方面可以加大探索力度，另一方面让监督信号的概率分布smooth一些，有点类似于监督学习里面的label smooth。
此外，对于根节点加入了0.25/0.75的狄利克雷噪声，非根节点加入0.1/0.9的狄利克雷噪声。在主游戏最终选择action的时候，只根据结点访问次数的概率分布选择action，不再加入噪声。



## 训练结果
上面的效果图是我训练到6960步的时候，人机对弈的结果。AI是黑方，我是白方（尽管最后还是我赢了），可以看到，AI在游戏前期还是不错的。会进攻会防守。但是后期就有点乏力了。
可能的原因是训练的次数不够，对弈后期的局面没有得到充分的训练。事实上，在6960步的时候，对弈的平均长度只有25步左右，需要继续往后训练，模拟对弈才会生成更长的对弈棋局。

各种loss如下图所示

![episode_length](https://ra), ![value_loss](https://ra), ![xentropy_loss](https://ra),
![entropy](https://ra), ![total_loss](https://ra),

图eposide_len反映的是模拟对弈的时候产生的棋局的步数，（不是回合数，黑方落子，白方再落子，即为2步）。从该图可以看出，在训练初期，由于是完全随机落子，棋局步数很长，达到了50+
然后随着训练进行，模拟对弈的棋局步数很快下降，说明AI逐渐掌握了游戏初步规则，需要把五颗棋子摆放成一条线才能赢，赶紧很快就摆成一条线了，这时候只知道进攻，不懂得防守。随着训练的继续进行，AI才逐渐知道怎么防守，游戏逐渐变长。也只有把前期的攻防都学好以后，游戏才能发展到后期。
从该图也可以看出，游戏的长度始终在增长，8k步的训练远远是不够的。

value_loss的变化图，曲线在7k步的时候突然增大了，原因是在7k步的时候把学习率由1e-3降为了2e-4。为啥学习率降低了会导致value loss陡增？有待研究。
所有的loss在8700步的时候都陡增了，是因为我这里掐断了，然后重新跑的。我明明同时保存了ckpt和data buffer，不知为啥依然会有这个现象。。不过loss下降的速度也很快。如果不用data buffer里面的数据，而是根据断点处重新生成数据，则新生成的数据都是基于断点处的network生成的，而network只对buffer里面的历史数据拟合得很好，对完全由断点处的network产生的数据拟合度未必好，产生loss陡增的现象还可以理解。
可能是代码哪里有bug。后续再研究。

entropy反映了输出policy的概率分布的熵。在初期随机落子，熵比较大，随着训练进行，熵自然就减小了。

图中显示的value loss和xentropy loss都是没有加权的，即棋局初期和后期的权重保持一致。而total loss是加权value loss，加权xentropy loss，L2正则化loss之和。


## 后续工作
1. 进一步往后训练。可能需要适当加大buffer size，并调整学习率；
2. 研究为何在7k步学习率下降会导致value loss陡增；
3. 研究为何掐断以后保持原来的数据和ckpt会导致loss陡增（虽然很快还是降下来了）。