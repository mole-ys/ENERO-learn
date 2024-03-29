# ENERO: Efficient real-time WAN routing optimization with Deep Reinforcement Learning

#### Link to paper: [here](https://www.sciencedirect.com/science/article/pii/S1389128622002717)

#### Paul Almasan, Shihan Xiao, Xiangle Cheng, Xiang Shi, Pere Barlet-Ros, Albert Cabellos-Aparicio

Contact: <felician.paul.almasan@upc.edu>

[![Twitter Follow](https://img.shields.io/twitter/follow/PaulAlmasan?style=social)](https://twitter.com/PaulAlmasan)
[![GitHub watchers](https://img.shields.io/github/watchers/BNN-UPC/ENERO?style=social&label=Watch)](https://github.com/BNN-UPC/ENERO)
[![GitHub forks](https://img.shields.io/github/forks/BNN-UPC/ENERO?style=social&label=Fork)](https://github.com/BNN-UPC/ENERO)
[![GitHub stars](https://img.shields.io/github/stars/BNN-UPC/ENERO?style=social&label=Star)](https://github.com/BNN-UPC/ENERO)

## Abstract
Wide Area Networks (WAN) are a key infrastructure in today’s society. During the last years, WANs have seen a considerable increase in network’s traffic and network applications, imposing new requirements on existing network technologies (e.g., low latency and high throughput). Consequently, Internet Service Providers (ISP) are under pressure to ensure the customer’s Quality of Service and fulfill Service Level Agreements. Network operators leverage Traffic Engineering (TE) techniques to efficiently manage the network’s resources. However, WAN’s traffic can drastically change during time and the connectivity can be affected due to external factors (e.g., link failures). Therefore, TE solutions must be able to adapt to dynamic scenarios in real-time.

In this paper we propose Enero, an efficient real-time TE solution based on a two-stage optimization process. In the first one, Enero leverages Deep Reinforcement Learning (DRL) to optimize the routing configuration by generating a long-term TE strategy. To enable efficient operation over dynamic network scenarios (e.g., when link failures occur), we integrated a Graph Neural Network into the DRL agent. In the second stage, Enero uses a Local Search algorithm to improve DRL’s solution without adding computational overhead to the optimization process. The experimental results indicate that Enero is able to operate in real-world dynamic network topologies in 4.5 s on average for topologies up to 100 links.

## Instructions to set up the Environment (It is recommended to use Linux)
This paper implements the PPO algorithm to train a DRL agent that learns to route src-dst traffic demands using middelpoint routing. 

1. First, create the virtual environment and activate the environment.
```ruby
virtualenv -p python3 myenv
source myenv/bin/activate
```

2. Then, we install all the required packages.
```ruby
pip install -r requirements.txt
```

3. Register custom gym environment.
```ruby
pip install -e gym-graph/
```

## Instructions to prepare the datasets

The source code already provides the data, the results and the trained model used in the paper. Therefore, we can start by using the datasets provided to obtain the figures used in the paper.

1. Download the dataset from [here](https://drive.google.com/file/d/1gem-VQ5MY3L54B77XUYt-rTbemyKmaqs/view?usp=sharing) or [here](https://bnn.upc.edu/download/enero-dataset/) and unzip it. The location should be immediatly outside of Enero's code directory. 

   从这里或这里下载数据集并解压缩。该位置应该在Enero的代码目录之外。

![image](https://user-images.githubusercontent.com/87467979/215685300-de8c071d-c8f7-4ffa-be6a-c642f04a7d76.png)

2. Then, enter in the unziped "Enero_datasets" directory and unzip everything.
  
   然后，进入解压后的“Enero_datasets”目录并解压所有内容。

## Instructions to obtain the Figures from the paper

1. First, we execute the following command:

```ruby
python figures_5_and_6.py -d SP_3top_15_B_NEW 
```

2. Then, we execute the following (one per topology):
```ruby
python figure_7.py -d SP_3top_15_B_NEW -p ../Enero_datasets/dataset_sing_top/evalRes_NEW_EliBackbone/EVALUATE/ -t EliBackbone

python figure_7.py -d SP_3top_15_B_NEW -p ../Enero_datasets/dataset_sing_top/evalRes_NEW_HurricaneElectric/EVALUATE/ -t HurricaneElectric

python figure_7.py -d SP_3top_15_B_NEW -p ../Enero_datasets/dataset_sing_top/evalRes_NEW_Janetbackbone/EVALUATE/ -t Janetbackbone
```

3. Next, we generate the link failure Figures (one per topology):

   接下来，我们继续对模型进行评估。例如，假设我们想在Garr199905拓扑上评估所提供的训练模型。要做到这一点，我们执行以下脚本，我们用标志'-d'表示选择训练模型，用标志'-f1'表示目录(它必须与前一个命令相同!)，用'-f2'指定拓扑。

```ruby
python figure_8.py -d SP_3top_15_B_NEW -num_topologies 20 -f ../Enero_datasets/dataset_sing_top/LinkFailure/rwds-LinkFailure_HurricaneElectric

python figure_8.py -d SP_3top_15_B_NEW -num_topologies 20 -f ../Enero_datasets/dataset_sing_top/LinkFailure/rwds-LinkFailure_Janetbackbone

python figure_8.py -d SP_3top_15_B_NEW -num_topologies 20 -f ../Enero_datasets/dataset_sing_top/LinkFailure/rwds-LinkFailure_EliBackbone
```

4. Finally, we generate the generalization figure

   最后，生成概化图

```ruby
python figure_9.py -d SP_3top_15_B_NEW -p ../Enero_datasets/rwds-results-1-link_capacity-unif-05-1-zoo
```

## Instructions to EVALUATE

To evaluate the model we should execute the following scripts. Each script should be executed independently for each topology where we want to evaluate the trained model. Notice that we should point to each topology were we want to evaluate with the flag -f2. In the paper, we evaluated on "NEW_EliBackbone", "NEW_Janetbackbone" and "NEW_HurricaneElectric".

为了评估模型，我们应该执行以下脚本。对于我们想要评估训练模型的每个拓扑，每个脚本都应该独立执行。注意，我们应该指向我们想用标志-f2求值的每个拓扑。本文对“NEW_EliBackbone”、“NEW_Janetbackbone”和“NEW_HurricaneElectric”进行了评价。

1. First of all, we need to split the original data from the desired topology between training and evaluation. To do this, we should choose from "../Enero_datasets/results-1-link_capacity-unif-05-1/results_zoo/" one topology that we want to evaluate on. Lets say we choose the Garr199905 topology. Then, we need to execute:

    首先，我们需要将原始数据从训练和评估的理想拓扑中分离出来。要做到这一点，我们应该从“……/Enero_datasets/results-1-link_capacity-unif-05-1/results_zoo/"我们想要评估的一个拓扑。假设我们选择Garr199905拓扑。然后，我们需要执行:

```ruby
python convert_dataset.py -f1 results_single_top -name Garr199905
```

2. Next, we proceed to evaluate the model. For example, let's say we want to evaluate the provided trained model on the Garr199905 topology. To do this we execute the followinG script were we indicate with the flag '-d' to select the trained model, with the flag '-f1' we indicate the directory (it has to be the same like in the previous command!) and with '-f2' we specify the topology.

   接下来，我们继续对模型进行评估。例如，假设我们想在Garr199905拓扑上评估所提供的训练模型。要做到这一点，我们执行以下脚本，我们用标志'-d'表示选择训练模型，用标志'-f1'表示目录(它必须与前一个命令相同!)，用'-f2'指定拓扑。

```ruby
python eval_on_single_topology.py -max_edge 100 -min_edge 5 -max_nodes 30 -min_nodes 1 -n 2 -f1 results_single_top -f2 NEW_Garr199905/EVALUATE -d ./Logs/expSP_3top_15_B_NEWLogs.txt
```

3. Once we evaluated over the desired topologies, we can plot the boxplot (Figures 5 and 6 from the paper). Before doing this, we should edit the script and make the "folder" list contain only the desired folder with the results of the previous experiments. Specifically, we edited folders like:

   一旦我们评估了所需的拓扑，我们就可以绘制箱线图(来自论文的图5和图6)。在此之前，我们应该编辑脚本，使“文件夹”列表中只包含包含前面实验结果的所需文件夹。具体来说，我们编辑的文件夹如下:

```ruby
folders = ["../Enero_datasets/dataset_sing_top/data/results_single_top/evalRes_NEW_Garr199905/EVALUATE/"]
```
In addition, we also need to modify the script to plot the boxplots properly. Then, we can execute the following command. If we evaluated our model on different topologies, we should modify the script and make the "folders" list include the proper directories.

此外，我们还需要修改脚本以正确绘制箱线图。然后，我们可以执行以下命令。如果我们在不同的拓扑结构上评估我们的模型，我们应该修改脚本并使“文件夹”列表包含适当的目录。

```ruby
python figures_5_and_6.py -d SP_3top_15_B_NEW 
```

4. We can obtain the Figure 7 from the paper executing the following script for each topology (i.e., "Garr199905").

```ruby
python figure_7.py -d SP_3top_15_B_NEW -p ../Enero_datasets/dataset_sing_top/data/results_single_top/evalRes_NEW_Garr199905/EVALUATE/ -t Garr199905
```


5. The next experiment would be the link failure scenario. To do this, we first need to generate the data with link failures. Specifically, we maintain the TMs but we remove links from the network.

   下一个实验是链路故障场景。为此，我们首先需要生成链路故障的数据。具体来说，我们维护TMs，但从网络中删除链接。

```ruby
python3 generate_link_failure_topologies.py -d results-1-link_capacity-unif-05-1 -topology Garr199905 -num_topologies 1 -link_failures 1
```

6. Now we already have the new topologies with link failures. Next is to execute DEFO on the new topologies. To do this, we need to edit the script *run_Defo_all_Topologies.py* and make it point to the new generated dataset. Then, execute the following command and run DEFO. Notice that with the '--optimizer' flag we can indicate to run other optimizers implemented in [REPETITA](https://github.com/svissicchio/Repetita).

   现在我们已经有了带有链路故障的新拓扑。接下来是在新的拓扑上执行DEFO。为此，我们需要编辑run_defo_all_topology .py脚本，并使其指向新生成的数据集。然后，执行以下命令并运行DEFO。注意，使用“——optimizer”标志，我们可以指示运行在REPETITA中实现的其他优化器。

```ruby
python3 run_Defo_all_Topologies.py -max_edges 80 -min_edges 20 -max_nodes 25 -min_nodes 5 -optim_time 10 -n 15 --optimizer 100
```

7. The next step is to evaluate the DRL agent on the new topologies. The following script will create the directory "rwds-LinkFailure_Garr199905" which is then used to create the figures.

   下一步是在新拓扑上评估DRL代理。下面的脚本将创建目录“rwds-LinkFailure_Garr199905”，然后使用该目录创建图形。

```ruby
python eval_on_link_failure_topologies.py -max_edge 100 -min_edge 2 -max_nodes 30 -min_nodes 1 -n 2 -d ./Logs/expSP_3top_15_B_NEWLogs.txt -f LinkFailure_Garr199905

python figure_7.py -d SP_3top_15_B_NEW -p ../Enero_datasets/dataset_sing_top/data/results_single_top/evalRes_LinkFailure_Garr199905/EVALUATE/ -t Garr199905
```

## Instructions to TRAIN

1. To trail the DRL agent we must execute the following command. Notice that inside the *train_Enero_3top_script.py* there are different hyperparameters that you can configure to set the training for different topologies, to define the size of the GNN model, etc. Then, we execute the following script which executes the actual training script periodically. 

   要跟踪DRL代理，我们必须执行以下命令。注意，在train_Enero_3top_script.py中有不同的超参数，您可以配置这些超参数来设置不同拓扑的训练，定义GNN模型的大小等。然后，我们执行下面的脚本，它定期执行实际的训练脚本。

```ruby
python train_Enero_3top.py
```

2. Now that the training process is executing, we can see the DRL agent performance evolution by parsing the log files from another terminal session. Notice that the following command should point to the proper Logs.

   现在训练过程正在执行，我们可以通过解析来自另一个终端会话的日志文件来查看DRL代理的性能演变。注意，下面的命令应该指向正确的日志。

```ruby
python parse_PPO.py -d ./Logs/expEnero_3top_15_B_NEWLogs.txt
```

Please cite the corresponding article if you use the code from this repository:

```
@article{ALMASAN2022109166,
title = {ENERO: Efficient real-time WAN routing optimization with Deep Reinforcement Learning},
journal = {Computer Networks},
volume = {214},
pages = {109166},
year = {2022},
issn = {1389-1286},
doi = {https://doi.org/10.1016/j.comnet.2022.109166},
url = {https://www.sciencedirect.com/science/article/pii/S1389128622002717},
author = {Paul Almasan and Shihan Xiao and Xiangle Cheng and Xiang Shi and Pere Barlet-Ros and Albert Cabellos-Aparicio},
keywords = {Routing, Optimization, Deep Reinforcement Learning, Graph Neural Networks},
abstract = {Wide Area Networks (WAN) are a key infrastructure in today’s society. During the last years, WANs have seen a considerable increase in network’s traffic and network applications, imposing new requirements on existing network technologies (e.g., low latency and high throughput). Consequently, Internet Service Providers (ISP) are under pressure to ensure the customer’s Quality of Service and fulfill Service Level Agreements. Network operators leverage Traffic Engineering (TE) techniques to efficiently manage the network’s resources. However, WAN’s traffic can drastically change during time and the connectivity can be affected due to external factors (e.g., link failures). Therefore, TE solutions must be able to adapt to dynamic scenarios in real-time. In this paper we propose Enero, an efficient real-time TE solution based on a two-stage optimization process. In the first one, Enero leverages Deep Reinforcement Learning (DRL) to optimize the routing configuration by generating a long-term TE strategy. To enable efficient operation over dynamic network scenarios (e.g., when link failures occur), we integrated a Graph Neural Network into the DRL agent. In the second stage, Enero uses a Local Search algorithm to improve DRL’s solution without adding computational overhead to the optimization process. The experimental results indicate that Enero is able to operate in real-world dynamic network topologies in 4.5 s on average for topologies up to 100 links.}
}
```
