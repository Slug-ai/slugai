from controller.scheduler import Scheduler


def arrays(frames):
    return [(msg, x.tolist()) for msg, x in frames]


def test_a_slot_starts_with_a_padded_prefill_then_the_last_tokens():
    s = Scheduler([[1, 2, 3], [4, 5], [6]], max_tokens=4, eos={0})
    assert arrays(s.start(0, 2)) == [
        ({"type": "forward", "slot": 0, "lengths": [2, 1]}, [[1, 2], [4, 0]]),
        ({"type": "forward", "slot": 0}, [[3], [5]]),
    ]
    assert arrays(s.start(1, 2)) == [({"type": "forward", "slot": 1, "lengths": [0]}, [[0]]), ({"type": "forward", "slot": 1}, [[6]])]
    assert s.start(2, 2) == []  # nothing left to start


def test_tokens_feed_the_next_step_until_every_row_ends():
    s = Scheduler([[1, 2], [3, 4], [5, 6]], max_tokens=3, eos={9})
    s.start(0, 2)
    assert arrays(s.advance(0, [7, 9])) == [({"type": "forward", "slot": 0}, [[7], [9]])]  # row 1 ended at eos
    s.advance(0, [8, 8])
    frames = s.advance(0, [8, 8])  # row 0 ended at max_tokens: the slot starts the next prompt
    assert [seq.tokens for seq in s.sequences[:2]] == [[7, 8, 8], [9]]
    assert arrays(frames)[0] == ({"type": "forward", "slot": 0, "lengths": [1]}, [[5]])
    s.advance(0, [9])
    assert s.finished and s.sequences[2].tokens == [9]


def test_requeued_sequences_resume_from_their_tokens():
    s = Scheduler([[1, 2], [3, 4], [5, 6]], max_tokens=5, eos={9})
    s.start(0, 2)
    s.advance(0, [7, 9])
    s.requeue()  # row 1 is done; row 0 goes back, ahead of prompt 2
    assert [seq.id for seq in s.waiting] == [0, 2] and not s.slots
    assert arrays(s.start(0, 1))[0] == ({"type": "forward", "slot": 0, "lengths": [2]}, [[1, 2]])  # context 1 2 7: prefill 1 2
