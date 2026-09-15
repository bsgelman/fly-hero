"""Songs, rewards and the millisecond-level episode runner. js/game.js is the browser port."""
from .sim import Rng

SLOT_MS, LEAD_MS, SLOTS = 400, 200, 48
WAIT = 3
# relative to each slot start (hit line at +200 ms)
T_DRIVE_ON, T_COUNT, T_DECIDE, T_LEARN, T_HIT = 0, 50, 240, 340, 200


def make_song(seed, name, slots=SLOTS):
    rand = Rng(seed)
    return {'name': name, 'notes': [-1 if rand() < 0.25 else int(rand() * 3) for _ in range(slots)]}


SONGS = {'train': make_song(101, 'Banana Drift'), 'test': make_song(202, 'Wing Hum (unseen)')}


def hit_time(slot):
    return LEAD_MS + slot * SLOT_MS + T_HIT


def song_ms(song):
    return LEAD_MS + len(song['notes']) * SLOT_MS


def reward(note, action):
    if note < 0:
        return 0 if action == WAIT else -0.5
    return 1 if action == note else -1


class Episode:
    """One play of a song, advanced 1 ms at a time."""

    def __init__(self, net, agent, song, learn=True):
        self.net, self.agent, self.song, self.learn = net, agent, song, learn
        self.t = 0
        self.hits = self.notes = self.wrong = self.misses = self.false_presses = 0
        self.events = []  # {slot, note, action, reward, t}
        self.end = song_ms(song)
        net.reset_state()

    @property
    def done(self):
        return self.t >= self.end

    @property
    def rate(self):
        return self.hits / self.notes if self.notes else 0

    def step(self):
        net, agent = self.net, self.agent
        local = self.t - LEAD_MS
        if local >= 0:
            slot, ph = divmod(local, SLOT_MS)
            note = self.song['notes'][slot]
            if ph == T_DRIVE_ON and note >= 0:
                net.set_drive(net.lane_vpn[note], 150)
            elif ph == T_COUNT:
                net.reset_counts()
            elif ph == T_DECIDE:
                if note >= 0:
                    net.set_drive(net.lane_vpn[note], 0)
                action = agent.decide(net)
                r = reward(note, action)
                if note >= 0:
                    self.notes += 1
                    if action == note:
                        self.hits += 1
                    elif action == WAIT:
                        self.misses += 1
                    else:
                        self.wrong += 1
                elif action != WAIT:
                    self.false_presses += 1
                self.events.append({'slot': slot, 'note': note, 'action': action, 'reward': r, 't': self.t})
                agent.reward(net, r)
                net.reset_counts()
            elif ph == T_LEARN:
                agent.after_dopamine(net, self.learn)
        net.step()
        self.t += 1

    def run(self):
        while not self.done:
            self.step()
        return self
