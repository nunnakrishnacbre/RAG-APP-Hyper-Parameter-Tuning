
import sqlite3
import hashlib
import traceback
import logs

from argon2 import hash_password


class CreateUserDB():
    def encrypttion_pwd(self,password):
        return hashlib.sha256(password.encode()).hexdigest()

    def setup_database(self):
        try:
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            cursor.execute('''
                    CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password TEXT,                
                    jobrole TEXT,
                    department TEXT,
                    location TEXT)
                    ''')
            cursor.execute("SELECT COUNT(*) FROM users")
            if cursor.fetchone()[0] == 0:
                users = [('admin', self.encrypttion_pwd('admin'), 'admin', 'admin', 'chennai'),
                    ('user1', self.encrypttion_pwd('user'), 'user', 'manager', 'chennai'),
                    ('user2', self.encrypttion_pwd('user'), 'user', 'pilot', 'chennai'),
                    ('user3', self.encrypttion_pwd('user'), 'user', 'engineer', 'chennai'),
                    ('user4', self.encrypttion_pwd('user'), 'user', 'cabin crew', 'chennai'),
                    ('user5', self.encrypttion_pwd('user'), 'user', 'ground staff', 'chennai')]
                cursor.executemany("INSERT INTO users VALUES(?,?,?,?,?)",users)
                conn.commit()
            else:
                logs.logger.info("Databases alerady contains users")

        except Exception as ex:
            traceback.print_exc()
            logs.logger.info(f"Exception occurred: {ex}")
        finally:
            if conn:
                conn.close()


if __name__ == "__main__":
    db_obj = CreateUserDB()
    db_obj.setup_database()
