import os
import asyncio
import time
import torch
import torch.nn.functional as F

from utils.utils import ExecutionTimer

from ..compute_loss import goal_curr_alive_mine_mask, append_loss, compute_gae, cal_hier_log_probs, rew_vec_to_scaled_scalar, cross_entropy_loss, Normalizier


Symlog = Normalizier.symlog
NormReturns = Normalizier.norm_returns
CalculateScale = Normalizier.calculate_s
TwohotDecoding = Normalizier.twohot_decoding
TwohotEncoding = Normalizier.twohot_encoding


async def learning(parent, timer: ExecutionTimer):
    assert hasattr(parent, "batch_queue")
    scale = parent.scale # 초기화
    
    while not parent.stop_event.is_set():
        batch_dict = None
        with timer.timer("learner-throughput", check_throughput=True):
            with timer.timer("learner-batching-time"):
                batch_dict = await parent.batch_queue.get()

            if batch_dict is not None:
                with timer.timer("learner-forward-time"):
                    # Basically, mini-batch-learning (batch, seq, feat)
                    assert "obs" in batch_dict
                    assert "act" in batch_dict
                    assert "rew" in batch_dict
                    assert "info" in batch_dict
        
                    obs_dict = {k: v.to(parent.device) for k, v, in batch_dict["obs"].items()}
                    act_dict = {k: v.to(parent.device) for k, v, in batch_dict["act"].items()}
                    rew_dict = {k: v.to(parent.device) for k, v, in batch_dict["rew"].items()}
                    info_dict = {k: v.to(parent.device) for k, v, in batch_dict["info"].items()}
                    
                    behav_log_probs  = cal_hier_log_probs(act_dict)
                    rew_sca = rew_vec_to_scaled_scalar(rew_dict)
                    
                    is_fir = info_dict["is_fir"]
                    hx, cx = obs_dict["hx"], obs_dict["cx"]
                    
                    # epoch-learning
                    for _ in range(parent.args.K_epoch):
                        # on-line model forwarding
                        log_probs, entropy, value = parent.model(
                            obs_dict,
                            act_dict,
                            hx[:, 0],
                            cx[:, 0],
                        )
                        with torch.no_grad():
                            v_res = TwohotDecoding(value, parent.model.bins)
                            
                            td_target_res = (
                                rew_sca[:, :-1]
                                + parent.args.gamma * (1 - is_fir[:, 1:]) * v_res[:, 1:]
                            )
                            delta = td_target_res - v_res[:, :-1]

                            # td_target_twohot = TwohotEncoding(Symlog(td_target_res), parent.model.bins)
                            td_target_twohot = TwohotEncoding(td_target_res, parent.model.bins)
                            
                            gae = compute_gae(
                                delta, parent.args.gamma, parent.args.lmbda
                            )  # ppo-gae (advantage)
                            
                            scale = CalculateScale(gae, previous_s=scale)
                            gae = NormReturns(gae, scale)
                            valid_mine_mask = goal_curr_alive_mine_mask(obs_dict, env_space=parent.env_space)
                            
                        ratio = torch.exp(
                            log_probs[:, :-1] - behav_log_probs[:, :-1]
                        )  # a/b == exp(log(a)-log(b))

                        surr1 = ratio * gae
                        surr2 = (
                            torch.clamp(
                                ratio,
                                1 - parent.args.eps_clip,
                                1 + parent.args.eps_clip,
                            )
                            * gae
                        )

                        loss_policy = -torch.min(surr1, surr2)[valid_mine_mask].mean()
                        # loss_value = F.smooth_l1_loss(value[:, :-1], td_target).mean()
                        loss_value = cross_entropy_loss(value[:, :-1], td_target_twohot)[valid_mine_mask.squeeze(-1)].mean()
                        policy_entropy = entropy[:, :-1][valid_mine_mask].mean()
                        
                        loss = append_loss(parent.args.policy_loss_coef * loss_policy)
                        loss = append_loss(parent.args.value_loss_coef * loss_value, loss)
                        loss = append_loss(-parent.args.entropy_coef * policy_entropy, loss)
                        
                        detached_losses = {
                            "policy-loss": loss_policy.detach().cpu(),
                            "value-loss": loss_value.detach().cpu(),
                            "policy-entropy": policy_entropy.detach().cpu(),
                            "scale": scale.item(),
                            "ratio": ratio.detach().cpu(),
                        }

                        with timer.timer("learner-backward-time"):
                            parent.optimizer.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                parent.model.parameters(),
                                parent.args.max_grad_norm,
                            )
                            print(
                                "loss: {:.5f} original_value_loss: {:.5f} original_policy_loss: {:.5f} original_policy_entropy: {:.5f} ratio-avg: {:.5f}".format(
                                    loss.item(),
                                    detached_losses["value-loss"],
                                    detached_losses["policy-loss"],
                                    detached_losses["policy-entropy"],
                                    detached_losses["ratio"].mean(),
                                )
                            )
                            parent.optimizer.step()

                # CPU 텐서로 변환 후 전송
                parent.pub_model({k: v.cpu() for k, v in parent.model.state_dict().items()})

                if parent.idx % parent.args.loss_log_interval == 0:
                    await parent.log_loss_tensorboard(timer, loss, detached_losses)

                if parent.idx % parent.args.model_save_interval == 0:
                    torch.save(
                        {
                            "model_state": parent.model.state_dict(),
                            "log_idx": parent.idx,
                            "scale": scale,
                            "optim_state_dict": parent.optimizer.state_dict(),
                        },           
                        os.path.join(
                            parent.args.model_dir, f"{parent.args.algo}_{parent.idx}.pt"
                        ),
                    )

                parent.idx += 1

            if parent.heartbeat is not None:
                parent.heartbeat.value = time.monotonic()

        await asyncio.sleep(1e-4)
