import os
import pickle

class UserManager:
    def __init__(self):
        self.user_dir = 'users'
        os.makedirs(self.user_dir, exist_ok=True)

    def save_user(self, username, model):
        with open(os.path.join(self.user_dir, f'{username}.pkl'), 'wb') as f:
            pickle.dump(model, f)

    def load_user(self, username):
        path = os.path.join(self.user_dir, f'{username}.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None

    def delete_user(self, username):
        path = os.path.join(self.user_dir, f'{username}.pkl')
        if os.path.exists(path):
            os.remove(path)

    def list_users(self):
        return [f.split('.')[0] for f in os.listdir(self.user_dir) if f.endswith('.pkl')]
