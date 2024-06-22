import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical, Normal


class MlpLSTMBase(nn.Module):
    def __init__(self, f, n_outputs, sequence_length, hidden_size):
        super().__init__()
        self.input_size = f
        self.n_outputs = n_outputs
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size

        self.body = nn.Sequential(
            nn.Linear(in_features=self.input_size, out_features=self.hidden_size),
            nn.ReLU(),
        )
        self.after_torso()

        self.CT = Categorical

    def after_torso(self):
        self.lstmcell = nn.LSTMCell(
            input_size=self.hidden_size, hidden_size=self.hidden_size
        )

        # value
        self.value = nn.Linear(in_features=self.hidden_size, out_features=1)

        # policy
        self.logits = nn.Linear(
            in_features=self.hidden_size, out_features=self.n_outputs
        )

    def act(self, obs, lstm_hxs):
        with torch.no_grad():
            x = self.body.forward(obs)  # x: (feat,)
            hx, cx = self.lstmcell(x, lstm_hxs)

            dist = self.get_dist(hx)
            action = dist.sample().detach()

            # TODO: 좀 이상한 코드..
            logits = (
                dist.logits.detach()
                if hasattr(dist, "logits")
                else torch.zeros(action.shape)
            )
        return (
            action,
            logits,
            dist.log_prob(action).detach(),
            (hx.detach(), cx.detach()),
        )

    def get_dist(self, x):
        logits = self.logits(x)
        probs = F.softmax(logits, dim=-1)  # logits: (batch, feat)
        return self.CT(probs)

    def forward(self, obs, lstm_hxs, behaviour_acts):
        batch, seq, *sha = obs.size()
        hx, cx = lstm_hxs

        obs = obs.contiguous().view(batch * seq, *sha)
        x = self.body.forward(obs)
        x = x.view(batch, seq, self.hidden_size)  # (batch, seq, hidden_size)

        output = []
        for i in range(seq):
            hx, cx = self.lstmcell(x[:, i], (hx, cx))
            output.append(hx)
        output = torch.stack(output, dim=1)  # (batch, seq, feat)

        value = self.value(output)  # (batch, seq, 1)
        dist = self.get_dist(output)  # (batch, seq, num_acts)

        if isinstance(dist, Categorical):
            behav_acts = behaviour_acts.squeeze(-1)
        else:
            assert isinstance(dist, Normal)
            behav_acts = behaviour_acts

        log_probs = dist.log_prob(behav_acts)
        entropy = dist.entropy()  # (batch, seq)

        # TODO: 좀 이상한 코드..
        logits = (
            dist.logits.view(batch, seq, -1)
            if hasattr(dist, "logits")
            else torch.zeros(behaviour_acts.shape)
        )
        return (
            logits,
            log_probs.view(batch, seq, -1),
            entropy.view(batch, seq, -1),
            value.view(batch, seq, -1),
        )


class MlpLSTMContinuous(MlpLSTMBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.NM = Normal

    def after_torso(self):
        super().after_torso()
        self.logits_std = nn.Linear(
            in_features=self.hidden_size, out_features=self.n_outputs
        )

    def get_dist(self, x):
        mu = torch.tanh(self.logits(x))  # mu
        std = F.softplus(self.logits_std(x))  # std
        # std = torch.clamp(log_std, min=-20, max=2).exp()
        return self.NM(mu, std)


class MlpLSTMActor(MlpLSTMBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def after_torso(self):
        self.lstmcell = nn.LSTMCell(
            input_size=self.hidden_size, hidden_size=self.hidden_size
        )

        # policy
        self.logits = nn.Linear(
            in_features=self.hidden_size, out_features=self.n_outputs
        )

    def forward(self, obs, lstm_hxs):
        batch, seq, *sha = obs.size()
        hx, cx = lstm_hxs

        obs = obs.contiguous().view(batch * seq, *sha)
        x = self.body.forward(obs)
        x = x.view(batch, seq, self.hidden_size)  # (batch, seq, hidden_size)

        output = []
        for i in range(seq):
            hx, cx = self.lstmcell(x[:, i], (hx, cx))
            output.append(hx)
        output = torch.stack(output, dim=1)  # (batch, seq, feat)

        logits = self.logits(output)  # (batch, seq, num_acts)
        probs = F.softmax(logits, dim=-1)  # (batch, seq, num_acts)

        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = probs == 0.0  # mask
        z = z.float() * 1e-8

        return (
            probs.view(batch, seq, -1),
            torch.log(probs + z).view(batch, seq, -1),
        )


class MlpLSTMActorContinuous(MlpLSTMContinuous):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def after_torso(self):
        self.lstmcell = nn.LSTMCell(
            input_size=self.hidden_size, hidden_size=self.hidden_size
        )

        # policy
        self.logits = nn.Linear(
            in_features=self.hidden_size, out_features=self.n_outputs
        )
        self.logits_log_std = nn.Linear(
            in_features=self.hidden_size, out_features=self.n_outputs
        )

    def act(self, obs, lstm_hxs):
        with torch.no_grad():
            x = self.body.forward(obs)  # x: (feat,)
            hx, cx = self.lstmcell(x, lstm_hxs)

            dist, action, log_prob = self.action_log_prob(hx)

            # TODO: 좀 이상한 코드..
            logits = (
                dist.logits.detach()
                if hasattr(dist, "logits")
                else torch.zeros(action.shape)
            )
        return (
            action.detach(),
            logits,
            log_prob.detach(),
            (hx.detach(), cx.detach()),
        )

    def get_dist(self, x):
        mu = self.logits(x)  # mu
        log_std = torch.clamp(self.logits_log_std(x), min=-20, max=2)  # std
        # std = torch.clamp(log_std, min=-20, max=2).exp()
        return self.NM(mu, log_std.exp())

    def action_log_prob(self, x):
        dist = self.get_dist(x)
        assert isinstance(dist, Normal)

        x_t = dist.rsample()
        action = torch.tanh(x_t)

        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-7)
        return dist, action, log_prob

    def forward(self, obs, lstm_hxs):
        batch, seq, *sha = obs.size()
        hx, cx = lstm_hxs

        obs = obs.contiguous().view(batch * seq, *sha)
        x = self.body.forward(obs)
        x = x.view(batch, seq, self.hidden_size)  # (batch, seq, hidden_size)

        output = []
        for i in range(seq):
            hx, cx = self.lstmcell(x[:, i], (hx, cx))
            output.append(hx)
        output = torch.stack(output, dim=1)  # (batch, seq, feat)

        _, action, log_prob = self.action_log_prob(output)
        return action.view(batch, seq, -1), log_prob.view(batch, seq, -1)


class MlpLSTMCritic(MlpLSTMBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # obs-encoding / overriding
        self.body = nn.Sequential(
            nn.Linear(in_features=self.input_size, out_features=self.hidden_size),
            nn.ReLU(),
        )

    def after_torso(self):
        self.lstmcell = nn.LSTMCell(
            input_size=self.hidden_size, hidden_size=self.hidden_size
        )

        # q-value
        self.q_value = nn.Linear(
            in_features=self.hidden_size, out_features=self.n_outputs
        )

    def forward(self, obs, lstm_hxs):
        batch, seq, *sha = obs.size()
        hx, cx = lstm_hxs

        obs = obs.contiguous().view(batch * seq, *sha)
        x_o = self.body.forward(obs)
        x_o = x_o.view(batch, seq, self.hidden_size)  # (batch, seq, hidden_size)

        output = []
        for i in range(seq):
            hx, cx = self.lstmcell(x_o[:, i], (hx, cx))
            output.append(hx)
        output = torch.stack(output, dim=1)  # (batch, seq, feat)

        q_value = self.q_value(output)  # (batch, seq, num_acts)

        return q_value.view(batch, seq, -1)


class MlpLSTMCriticContinuous(MlpLSTMBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # obs-encoding / overriding
        self.body = nn.Sequential(
            nn.Linear(
                in_features=self.input_size, out_features=int(self.hidden_size / 2)
            ),
            nn.ReLU(),
        )

    def after_torso(self):
        self.lstmcell = nn.LSTMCell(
            input_size=self.hidden_size, hidden_size=self.hidden_size
        )
        self.encode_action = nn.Sequential(
            nn.Linear(in_features=1, out_features=int(self.hidden_size / 2)),
            nn.ReLU(),
        )
        # q-value
        self.q_value = nn.Linear(in_features=self.hidden_size, out_features=1)

    def forward(self, obs, act, lstm_hxs):
        batch, seq, *sha = obs.size()
        hx, cx = lstm_hxs

        obs = obs.contiguous().view(batch * seq, *sha)
        x_o = self.body.forward(obs)
        x_o = x_o.view(
            batch, seq, int(self.hidden_size / 2)
        )  # (batch, seq, hidden_size/2)

        act = act.contiguous().view(batch * seq, -1)
        x_a = self.encode_action.forward(act)
        x_a = x_a.view(
            batch, seq, int(self.hidden_size / 2)
        )  # (batch, seq, hidden_size/2)

        x = torch.cat([x_o, x_a], -1)

        output = []
        for i in range(seq):
            hx, cx = self.lstmcell(x[:, i], (hx, cx))
            output.append(hx)
        output = torch.stack(output, dim=1)  # (batch, seq, feat)

        q_value = self.q_value(output)  # (batch, seq, num_acts)

        return q_value.view(batch, seq, -1)


class MlpLSTMDoubleCriticContinuous(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.q1 = MlpLSTMCriticContinuous(*args, **kwargs)
        self.q2 = MlpLSTMCriticContinuous(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.q1(*args, **kwargs), self.q2(*args, **kwargs)


class MlpLSTMDoubleCritic(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.q1 = MlpLSTMCritic(*args, **kwargs)
        self.q2 = MlpLSTMCritic(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.q1(*args, **kwargs), self.q2(*args, **kwargs)


class MlpLSTMSingle(nn.Module):
    """Actor, Critic이 하나의 torso"""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.actor = MlpLSTMBase(*args, **kwargs)
        self.critic = self.actor  # TODO: 개선 필요.. 동일 메모리 주소 참조 / dummy-code


class MlpLSTMSingleContinuous(nn.Module):
    """Actor, Critic이 하나의 torso"""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.actor = MlpLSTMContinuous(*args, **kwargs)
        self.critic = self.actor  # TODO: 개선 필요.. 동일 메모리 주소 참조 / dummy-code


class MlpLSTMSeperate(nn.Module):
    """Actor, Critic이 별도의 분리된 torso"""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.actor = MlpLSTMActor(*args, **kwargs)
        self.critic = MlpLSTMDoubleCritic(*args, **kwargs)


class MlpLSTMSeperateContinuous(nn.Module):
    """Actor, Critic이 별도의 분리된 torso"""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.actor = MlpLSTMActorContinuous(*args, **kwargs)
        self.critic = MlpLSTMDoubleCriticContinuous(*args, **kwargs)
