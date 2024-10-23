from state import State

class RandomAgent:
    async def select_move_index(self, state: State):
        return state.select_random_valid_move_index()
