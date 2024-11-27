import time
import math
import random
import torch
from torch import nn

def feed_window_until_amount(game_data_manager, amount_to_feed, new_data_check_interval):
    while True:
        samples_fed = game_data_manager.cumulative_window_fed()
        gap = max(amount_to_feed - samples_fed, 0)
        game_data_manager.feed_window(gap)

        if game_data_manager.cumulative_window_fed() >= amount_to_feed:
            break
        else:
            time.sleep(new_data_check_interval)

    assert game_data_manager.cumulative_window_fed() == amount_to_feed

def loop_iteration(
    model,
    optimizer,
    game_data_manager,
    *,
    device,
    batch_size,
    sampling_ratio,
    policy_loss_weight,
):
    # E.g. if sampling ratio is 2, then we need to read half a batch size before 
    # we can train on the batch we just read. The floor/random bits ensure we round
    # correctly, e.g. 2.1 rounds to 2 90% of the time and 3 10% of the time.
    required_new_samples = int(math.floor(batch_size / sampling_ratio + random.random()))

    ingestion_count = game_data_manager.feed_window(required_new_samples, require_exact_count=True)

    # Not enough new data to train a batch. Try again later.
    if not ingestion_count:
        return

    # Great, we've loaded enough data to train a new batch. Now, train on it.
    boards, policies, final_game_values, average_rollout_values = game_data_manager.sample(batch_size)

    boards = boards.to(dtype=torch.float32, device=device)
    policies = policies.to(dtype=torch.float32, device=device)
    final_game_values = final_game_values.to(dtype=torch.float32, device=device)
    average_rollout_values = average_rollout_values.to(dtype=torch.float32, device=device)

    assert len(boards) == batch_size

    values = (final_game_values + average_rollout_values) / 2

    pred_values, pred_policy = model(boards)
    value_loss = nn.CrossEntropyLoss()(
        pred_values,
        values,
    )
    policy_loss = policy_loss_weight * nn.CrossEntropyLoss()(
        pred_policy,
        policies,
    )
    loss = value_loss + policy_loss

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return {
        "value_loss": value_loss.item(),
        "policy_loss": policy_loss.item(),
        "loss": loss.item(),
        "batch_size": len(boards),
        "cumulative_window_fed": game_data_manager.cumulative_window_fed(),
    }
