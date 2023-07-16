#!/bin/bash


#========================================= cora ================================================
# python main.py -data cora -tag dir -m gcn -algo FedAvg -lr 0.05 -ls 5 -gr 100 -nc 5 -nb 7
# python main.py -data cora -tag dir -m gcn -algo Ditto -lr 0.05 -ls 5 -gr 100 -nc 5 -nb 7 -pls 2 -mu 0.5
# python main.py -data cora -tag dir -m gcn -algo FedAMP -lr 0.05 -ls 5 -gr 100 -nc 5 -nb 7 -alk 5e-3 -lam 5e-7 -sg 1e-1
# python main.py -data cora -tag dir -m gcn -algo pFedMe -lr 0.05 -ls 5 -gr 100 -nc 5 -nb 7 -lrp 0.09 -bt 1 -lam 15
# python main.py -data cora -tag dir -m gcnper -algo FedPer -lr 0.05 -ls 5 -gr 100 -nc 5 -nb 7
# python main.py -data cora -tag dir -m gcnper -algo FedRep -lr 0.05 -ls 5 -gr 100 -nc 5 -nb 7 -pls 10
# python main.py -data cora -tag dir -m grace -algo FedAvg -lr 0.05 -ls 5 -gr 100 -nc 5 -nb 7 --ssl_enabled --param cora.json
# python main.py -data cora -tag dir -m grace -algo Ditto -lr 0.05 -ls 5 -gr 100 -nc 5 -nb 7 -pls 10 -mu 0.5 --ssl_enabled --param cora.json


#======================================= citeseer =============================================
# python main.py -data citeseer -tag dir -m gcn -algo FedAvg -lr 0.05 -ls 5 -gr 100 -nc 5 -nb 6
# python main.py -data citeseer -tag dir -m gcn -algo Ditto -lr 0.1 -ls 5 -gr 100 -nc 5 -nb 6 -pls 5 -mu 0.5
# python main.py -data citeseer -tag dir -m grace -algo FedAvg -lr 0.1 -ls 5 -gr 100 -nc 5 -nb 6 --ssl_enabled --param citeseer.json



#======================================== pubmed ================================================
# python main.py -data pubmed -tag metis -m gcn -algo FedAvg -lr 0.05 -ls 5 -gr 100 -nc 5 -nb 5
# 




#========================================== CS =====================================================
# python main.py -data CS -tag metis -m gcn -algo FedAvg -lr 0.05 -ls 5 -gr 100 -nc 10 -nb 15
# python main.py -data CS -tag metis -m gcn -algo Ditto -lr 0.05 -ls 5 -gr 100 -nc 10 -nb 15 -pls 5 -mu 0.5
# python main.py -data CS -tag metis -m gcn -algo FedAvg -lr 0.05 -ls 5 -gr 100 -nc 10 -nb 15 -pls 5
# python main.py -data CS -tag metis -m grace -algo FedAvg -lr 0.05 -ls 5 -gr 100 -nc 10 -nb 15 --ssl_enabled --param coauthor_cs.json
# python main.py -data CS -tag metis -m grace -algo Ditto -lr 0.05 -ls 5 -gr 100 -nc 10 -nb 15 -pls 5 -mu 0.5 --ssl_enabled --param coauthor_cs.json
# python main.py -data CS -tag metis -m grace -algo FedPer -lr 0.05 -ls 5 -gr 100 -nc 10 -nb 15 --ssl_enabled --param coauthor_cs.json
# python main.py -data CS -tag metis -m grace -algo FedRep -lr 0.05 -ls 5 -gr 100 -nc 10 -nb 15 -pls 10 -ld --ssl_enabled --param coauthor_cs.json
# python main.py -data CS -tag metis -m gcnper -algo FedPer -lr 0.05 -ls 5 -gr 100 -nc 10 -nb 15
# python main.py -data CS -tag metis -m gcnper -algo FedRep -lr 0.05 -ls 5 -gr 100 -nc 10 -nb 15 -pls 10


#========================================== Physics ================================================
# python main.py -data Physics -tag dir -m gcn -algo FedAvg -lr 0.05 -ls 5 -gr 100 -nc 10 -nb 5
# python main.py -data Physics -tag dir -m gcn -algo Ditto -lr 0.05 -ls 5 -gr 100 -nc 10 -nb 5 -pls 5 -mu 0.5

#========================================== Cora-finetune ==========================================
# python main.py -data cora -out cora_ssl -tag ### -m grace -algo Ditto -lr 0.05 -ls 5 -gr 100 -nc 5 -nb 7 -pls 5 -mu 0.5 --ssl_enabled --param cora.json

#========================================== Physics-finetune =======================================
# python main.py -data Physics -out phy_ssl -tag dir -m grace -algo Ditto -lr 0.05 -ls 5 -gr 100 -nc 10 -nb 5 -pls 5 -mu 0.5 --ssl_enabled --param coauthor_phy.json