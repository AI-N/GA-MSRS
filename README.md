# GA-MSRS
In this work, we propose a multi-stakeholder recommender system using multi-objective evolutionary algorithm.
There are three main objectives in this work as follows: 
(1) accuracy, 
(2) the inclusion of long-tail items, 
(3) provider fairness. 
Our aim is to satisfy both users and providers in a recommender system. To do so, we propose an algorithm to include more niche items in a fair manner towards providers, while the accuracy is almost kept. As the objectives run counter each other, our problem is a multi-objective optimization problem. So, we solve the problem with NSGA-II, which is a Multi-Objective Evolutionary Algorithm (MOEA). NSGA-II is based on the concept of Pareto optimality. therefore the main goal is to find Pareto Front (PF), which is a set of recommendation list that make a trade-off among objective functions. Finally, each user can find his/her desired items in PF. 
We run the experiments on the proposed method and some existing works for two real-world datasets of movielens. Our method shows better provider coverage and long-tail coverage with a minor loss in accuracy.
