# mcts
A simple vanilla Monte-Carlo Tree Search implementation in python 

TODO : 

1. The MCTS logic is now using neural network as its simulation (evaluation or rollout function) backbone engine, will consider to use Deep Q-learning method later

2. Investigate the [PUCT formula](https://slides.com/crem/lc0#/9) more in-depth especially [Hoeffding’s Inequality](https://lilianweng.github.io/lil-log/2018/01/23/the-multi-armed-bandit-problem-and-its-solutions.html#hoeffdings-inequality) and its [weakness](https://hal.archives-ouvertes.fr/hal-00747575v4/document#page=27).  Study the performance quality of exploration/exploitation solved by PUCT which is mathematically represented by [Regret analysis](https://tor-lattimore.com/downloads/talks/2018/trieste/tr1.pdf#page=18)

3. Review the BAI-MCTS paper : [Monte-Carlo Tree Search by Best Arm Identification](https://arxiv.org/abs/1706.02986) , and [Adversarial Bandit Environments](https://tor-lattimore.com/downloads/book/book.pdf#page=156) 

Credit: Thanks to kind folks ([@crem](https://github.com/mooskagh) , [@Naphthalin](https://github.com/Naphthalin) , [@oscardssmith](https://github.com/oscardssmith)) from Leela-Zero community , and [@Lattimore](https://github.com/tor) from DeepMind
