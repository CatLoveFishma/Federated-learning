#1.每个数据集中的DE长度不一样，导致最后组合成的总的DE长度差距太大
    for iter in range(args.epochs): #循环epoch次
        t2=time.perf_counter()
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1) #挑选出客户的数量，frac是客户比例默认0.1，num_users是客户数量，默认100
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        #随机选客户，客户的index是[0,num_users-1]，num_users.default=100，选出的个数是m个，不可重复
        for idx in idxs_users: #循环遍历选出的客户
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            #本地更新？创建一个对象，传递训练数据集，
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            #net是将配置好的网络net_glob传递进去
            #to(args.device)指定训练时候是在cpu训练还是在gpu训练
            #调用local对象里面的训练函数，并且返回梯度w，和loss值
            #return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
            #w是net.state_dict()是一个字典包含训练后得到的权重，loss平均loss值
            if args.all_clients: #如果是训练所有客户，args.all_clients ，action="store_true"，默认false
                w_locals[idx] = copy.deepcopy(w) #更新w_locals中index=idx的客户的w
            else:   #如果不是训练所有客户,默认训练10个客户
                w_locals.append(copy.deepcopy(w))   #直接在w_locals添加。因为若不是所有客户都训练，w_locals被定义为[],空list
            loss_locals.append(copy.deepcopy(loss)) #将训练返回的loss值添加进loss_locals，每次训练添加一个loss值
        # update global weights
        w_glob = FedAvg(w_locals)
        #调用聚合函数 FedAvg ，传入的w_locals是返回的客户机的state_dict所构成的list

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        #torch.load_state_dict()函数就是用于将预训练的参数权重加载到新的模型之中,更新服务器的模型
        # print loss
        t3=time.perf_counter()
        all_time.append(t3-t2)
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f},训练耗时 {}s'.format(iter, loss_avg,(t3-t2)))
        loss_train.append(loss_avg)
