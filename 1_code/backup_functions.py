def analyze_model(test_loader, num_node, model_index, device, gnn_layers,
                  eff_model=None, vout_model=None, eff_vout_model=None, reward_model=None, cls_vout_model=None):
    """
    Find the optimal simulator reward of the topologies with the top-k surrogate rewards.
    """
    n_batch_test = 0

    sim_rewards = []
    gnn_rewards = []

    all_sim_eff = []
    all_sim_vout = []
    all_gnn_eff = []
    all_gnn_vout = []

    test_size = len(test_loader) * 256
    print("Test bench size:", test_size)

    k_list = [int(test_size * 0.01 + 1), int(test_size * 0.05 + 1), int(test_size * 0.1 + 1), int(test_size * 0.5 + 1)]

    for data in test_loader:
        # load data in batches and compute their surrogate rewards
        data.to(device)
        L = data.node_attr.shape[0]
        B = int(L / num_node)
        node_attr = torch.reshape(data.node_attr, [B, int(L / B), -1])
        if model_index == 0:
            edge_attr = torch.reshape(data.edge0_attr, [B, int(L / B), int(L / B), -1])
        else:
            edge_attr1 = torch.reshape(data.edge1_attr, [B, int(L / B), int(L / B), -1])
            edge_attr2 = torch.reshape(data.edge2_attr, [B, int(L / B), int(L / B), -1])

        adj = torch.reshape(data.adj, [B, int(L / B), int(L / B)])

        sim_eff = data.sim_eff.cpu().detach().numpy()
        sim_vout = data.sim_vout.cpu().detach().numpy()

        n_batch_test = n_batch_test + 1
        if eff_vout_model is not None:
            # using a model that can predict both eff and vout
            out = eff_vout_model(input=(
                node_attr.to(device), edge_attr1.to(device), edge_attr2.to(device), adj.to(device),
                gnn_layers)).cpu().detach().numpy()
            gnn_eff, gnn_vout = out[:, 0], out[:, 1]

        elif reward_model is not None:
            out = reward_model(input=(
                node_attr.to(device), edge_attr1.to(device), edge_attr2.to(device), adj.to(device),
                gnn_layers)).cpu().detach().numpy()
            all_sim_eff.extend(sim_eff)
            all_sim_vout.extend(sim_vout)
            sim_rewards.extend(compute_batch_reward(sim_eff, sim_vout))
            gnn_rewards.extend(out[:, 0])

            # all_* variables are updated here, instead of end of for loop
            # todo refactor
            continue

        elif cls_vout_model is not None:
            eff = eff_model(input=(node_attr.to(device), edge_attr1.to(device), edge_attr2.to(device), adj.to(device),
                                   gnn_layers)).cpu().detach().numpy()
            vout = cls_vout_model(input=(
                node_attr.to(device), edge_attr1.to(device), edge_attr2.to(device), adj.to(device),
                gnn_layers)).cpu().detach().numpy()

            gnn_eff = eff.squeeze(1)
            gnn_vout = vout.squeeze(1)
            all_sim_eff.extend(sim_eff)
            all_sim_vout.extend(sim_vout)
            all_gnn_eff.extend(gnn_eff)
            all_gnn_vout.extend(gnn_vout)

            tmp_gnn_rewards = []
            for j in range(len(gnn_eff)):
                tmp_gnn_rewards.append(gnn_eff[j] * gnn_vout[j])

            sim_rewards.extend(compute_batch_reward(sim_eff, sim_vout))
            gnn_rewards.extend(tmp_gnn_rewards)
            continue

        elif model_index == 0:
            eff = eff_model(input=(node_attr.to(device), edge_attr.to(device), adj.to(device))).cpu().detach().numpy()
            vout = vout_model(input=(node_attr.to(device), edge_attr.to(device), adj.to(device))).cpu().detach().numpy()

            gnn_eff = eff.squeeze(1)
            gnn_vout = vout.squeeze(1)
        else:
            eff = eff_model(input=(node_attr.to(device), edge_attr1.to(device), edge_attr2.to(device), adj.to(device),
                                   gnn_layers)).cpu().detach().numpy()
            vout = vout_model(input=(node_attr.to(device), edge_attr1.to(device), edge_attr2.to(device), adj.to(device),
                                     gnn_layers)).cpu().detach().numpy()

            gnn_eff = eff.squeeze(1)
            gnn_vout = vout.squeeze(1)

        all_sim_eff.extend(sim_eff)
        all_sim_vout.extend(sim_vout)
        all_gnn_eff.extend(gnn_eff)
        all_gnn_vout.extend(gnn_vout)

        sim_rewards.extend(compute_batch_reward(sim_eff, sim_vout))
        gnn_rewards.extend(compute_batch_reward(gnn_eff, gnn_vout))
        # out_list.extend(r)

    for k in k_list:
        print('k', k)
        print('Top-k topology analysis:')
        print(evaluate_top_K(gnn_rewards, sim_rewards, k))
        print('Bottom-k topology analysis:')
        print(evaluate_bottom_K(gnn_rewards, sim_rewards, k))
        # gnn_coverage[k] = top_K_coverage_on_ground_truth(gnn_rewards, sim_rewards, k, k)
