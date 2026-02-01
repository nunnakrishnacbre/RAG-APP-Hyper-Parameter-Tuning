import logs
import inspect
import traceback
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
from configuration import Configuration
class HostingInHuggingFace:
  def __init__(self):
    self.base_path = Configuration.PROJECT_ROOT
    self.hf_token = Configuration.HF_TOKEN
    self.repo_id = 'jpkarthikeyan/FlyKiteAirlines'

  def CreatingSpaceInHF(self):
    logs.logger.info(f"Function Name {inspect.currentframe().f_code.co_name}")
    api = HfApi()
    try:
      logs.logger.info(f"Checking for {self.repo_id} is correct or not")
      api.repo_info(repo_id = self.repo_id,
                    repo_type='space',
                    token = self.hf_token)
      logs.logger.info(f"Space {self.repo_id} already exists")
    except RepositoryNotFoundError:
      create_repo(repo_id=self.repo_id,
                  repo_type='space',
                  space_sdk='docker',
                  private=False,
                  token=self.hf_token)
      logs.logger.info(f"Space created in {self.repo_id}")
    except Exception as ex:
      logs.logger.info(f"Exception in creating space {ex}")
      logs.logger.info(traceback.print_exc())
    finally:
      logs.logger.info('-'*50)


  def UploadDeploymentFile(self):
    logs.logger.info(f"Function Name {inspect.currentframe().f_code.co_name}")
    try:
      api = HfApi(token=self.hf_token)
      directory_to_upload = self.base_path
      logs.logger.info(f"Directory to upload {directory_to_upload} into HF Space {self.repo_id}")
      api.upload_folder(repo_id=self.repo_id, folder_path=directory_to_upload,
                        repo_type='space')
      logs.logger.info(f"Successfully upload {directory_to_upload} into {self.repo_id}")
      return True
    except Exception as ex:
      logs.logger.info(f"Exception occured {ex}")
      logs.logger.info(traceback.print_exc())
      return False
    finally:
      logs.logger.info('-'*50)

  def ToRunPipeline(self):
    try:
      self.CreatingSpaceInHF()
      if self.UploadDeploymentFile():
        logs.logger.info('Deployment pipeline completed')
        return True
      else:
        logs.logger.info('Deployment pipeline failed')
        return False
    except Exception as ex:
      logs.logger.info(f"Exception occured {ex}")
      logs.logger.info(traceback.print_exc())
      return False
    finally:
      logs.logger.info('-'*50)


if __name__ == '__main__':
    hosting = HostingInHuggingFace()
    hosting.ToRunPipeline()
