# HumanFirst custom integration
This repository contains custom integration code that can be used to connect various NLU platforms to a HumanFirst namespace in order to extend its functionality.

Custom integrations are a set of GRPC services that can be implemented by a third party in order to provide extensibility, there are currently two types of services that can be implemented: Workspace and Model.

### Workspace
A workspace integration allows importing and exporting data. Once a custom integration with workspace support is added, it appears in the "Import > From Integration" and "Export > From Integration" workflows, allowing the users to import/export directly from the user interface.

Typically, a such service would merge a workspace's data before exporting a workspace (HF only deals with training data, anything else needs to be reconciliated). While you can use any supported data format, using [HumanFirst JSON](https://docs.humanfirst.ai/docs/advanced/humanfirst-json/) is recommended because it is the most feature-complete export format available.

#### Import
The import-related methods in the service are called whenever an export is done from the application's perspective.

- The `ListWorkspaces` method is called so the user can select which workspace to target
- The `GetImportParameters` method is called in order for the `Import` call to receive data in the right format
- The `Import` method is called with the exported data in the right format

#### Export
The export-related methods in the service are called whenever an import is performed from the application's perspective.

- The `ListWorkspaces` method is called so the user can select which workspace to export
- The `Export` method is called, which returns the exported data, along with information around its data format.

### Model
**NOTE: Currently this is work-in-progress**

A model integration allows training and evaluating NLU models, and optionally provide support for custom embedding models. The same abstraction handles both kinds of model since it's typical to use an embedding space as an intermediate latent representation inside a classification objective. 


## Hosting custom integration in GCP
How to start an instance to run custom integration in GCP?
* Go to GCP project -> Compute engine -> VM instance -> Creeate new VM instance
* Machine Configuration E2
* Machine type e2-medium (2vcpu,1 core, 1 4GB memory RAM)
* VM provisioning model -> standard
* Boot disk -> change 
    * OS -> Ubuntu
    * Version -> Ubuntu 20.04 LTS
    * Boot Disk type -> Balanced Persistent disk
    * Size -> 25 GB
* Access scopes -> Allow default access
* Firewall
    * Enable
        * Allow HTTP traffic
        * Allow HTTPS traffic
        * Allow Load Balancer Health Checks
* Advanced options -> Networking -> network interfaces
    * IP stack type -> IPv4
    * Primary internal IPv4 address -> Ephemeral
    * Externale IPv4 Address -> Either Ephemeral or can create a static IP address and use that here
* Then create
* Check if the instacne is started. If not, then start manually.

## Connect to instance from visual studio code
**Note: This is for windows**

* Create a public and private key
    * Go to powershell and execute the following command
        ```
        ssh-keygen -t rsa -f C:\Users\<windows username>\.ssh\<filename> -C <username> -b 2048
        cd C:\Users\<windows username>\.ssh\
        cat <filename>.pub
        ```
    * Copy the public key contents
* Go to Compute engine and under settings -> metadata -> SSH keys -> Edit -> Add the copied public key
* This gives project wide access to all the instance
* If you want to get access to only a particualr instance, then open that instance -> Edit ->  SSH keys -> Add item -> Paste the SSH key
* Open VS code in local system
* Press F1
* Search for "Remote-SSH: Connect to Host"
* Configure SSH Hosts
* Click/Go to C:\Users\<username>\.ssh\config
    ```
    Host <any name>
        HostName <VM instance external IP address>
        User <username>
        IdentityFile C:\Users\<windows username>\.ssh\<filename>
    ```
Then again click F1 -> "Remote-SSH: Connect to Host" -> Click host name

## Clone the repo

```
git clone https://github.com/zia-ai/hf-custom-integration.git
cd hf-custom-integration/
cp .bashrc_custom ~/.bashrc
source ~/.bashrc
```
Enable Pylance, Go and other entensions in VS code that you feel necessary

## Creating a custom integration
Custom integrations use GRPC with Mutual TLS authentication. The [hf](https://github.com/zia-ai/humanfirst/releases?q=cli&expanded=true) command-line tool contains a command allowing you to create an integration and generate MTLS credentials simultaneously.

1. Download the [hf](https://github.com/zia-ai/humanfirst/releases?q=cli&expanded=true) command line tool
Example command installing CLI-1.35.0
    ```
    sudo wget https://github.com/zia-ai/humanfirst/releases/download/cli-1.35.0/hf-linux-amd64?raw=true -O /usr/local/bin/hf && sudo chmod 755 /usr/local/bin/hf
    ```
2. [Authenticate](https://docs.humanfirst.ai/docs/cli/overview#authenticating) using your HF account
3. Set your namespace `hf namespace use <namespace>`
4. Create the integration
    ```
    hf integrations create custom --workspace --name <custom-integration-name> --address <dns name/Publicc IP address>:443 --key-out ./credentials/mtls-credentials.json --model
    ```

    Example:
    ```
    hf integrations create custom --workspace --name cust-intg-99 --address 31.218.151.124:443 --key-out ./credentials/mtls-credentials.json --model
    ```
5. Install all the dependencies: 
    ```
    sudo apt-get update
    sudo apt-get install python3.8-venv
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install poetry==1.3.2 --no-cache
    poetry install
    ```
6. As we are going to use the port 443, make sure to convert it into an unpreviledged port using `sudo sysctl -w net.ipv4.ip_unprivileged_port_start=443`
7. Get the Azure endpoint and API key using https://portal.azure.com/
    Go to resource
    Find Endpoint and keys
    
8. Set HumanFirst environment variables
    ```
    export HF_USERNAME="<HumanFirst Username>"
    export HF_PASSWORD="<HumanFirst Password>"
    ```
9. Set environment variables for running CLU integration
    ```
    export CLU_ENDPOINT="<CLU Endpoint>"
    export CLU_KEY="<CLU API key>"
    ```
**Note: In case of restarting the instance, ensure to run the follwoing command again - `sudo sysctl -w net.ipv4.ip_unprivileged_port_start=443`**

10. Launch the integration service:
    ```
    poetry run python3 -m hf_integration.main ./credentials/mtls-credentials.json 0.0.0.0:443 <integration-generic,clu,example> "<config - key1::value1,key2::value2,..,keyN::valueN>"
    ```

    Example:
    ```
    poetry run python3 -m hf_integration.main ./credentials/my_org-mtls-credentials.json 0.0.0.0:443 clu "clu_endpoint::$CLU_ENDPOINT,clu_key::$CLU_KEY,delimiter::-,project_path::/home/FayazJelani/hf-custom-integration,clu_language::ja,clu_multilingual::True,clu_training_mode::advanced,log_level::debug"
    ```

11. IF the IP address of the integration server changes, then use the following command to set the IP address of the integration server in the HF
`hf integrations --id intg-id-here set-address -a <Public IP Address>:443`

## Docker
**Note: This does not work properly. This is still a work-in-progress**
You can also build a docker container for the integration and launch it directly:

```
docker build -t hf-integration .
docker run -it --rm -v $(pwd):/src -p 443:443 hf-integration ./mtls-credentials.json 0.0.0.0:443 <integration-generic,clu,example> "<config - key1::value1,key2::value2,..,keyN::valueN>"
```

## Set up custom NLU
**Note: Currently this can be done only by any HF team**

Follow the steps here - https://www.notion.so/humanfirst/Custom-NLU-d4bb84f086764e8789c57c0b77a0fdeb#e0c9cbd9a50146589a37e82cddaf9440

**Notes:**

**1. Merge while importing into HF tool works only for addition of intents, entities, tags, phrases. Updation and deletion of existing phrases/intent/entities in the CLU are not reflected in the merge process**

**2. Export from HF to any NLU platfrom, would completely overwrite the existing information in the NLU platform**

**3. It is suggested to use clear everything before import from any NLU platform**

**4. Tags in HF apart from Train and Test, when importing from CLU platform gets removed when followiing step3**

**5. CLU JSON does not have IDs for intents, entities and training/testing phrases. So this makes it difficult to get changes made even using any custom scripts in the backend**

**6. Tags in HF workspace, upon exporting the workspace to CLU do not cause any changes in CLU except for the Train and Test tags used in labelled phrases**

**7. Integration is created under assumption that entities in MS CLU workspace is both learned and list to make the entity mapping work seamlessly. Not all clients will be willing to annotate all the list entities in MS CLU workspace. An easy way is to bring the workspace into HF and use Find annotations button and annotate everything and push it back to CLU**

**8. The integration would skip empty intents and takes into account only that has training phrases in them**

**9. Specifying the delimiter in GUI whie exporting doesn't do anything.**

**10. Set the delimiter right when you start the integration**

**11. If the evaluation in HF side performs retry logic, then it does not send any request to delete the agent which was having issues it created in the clu**

**12. Before running NLU training, ensure to perform bi-directional sync.**

**13. Remove train tag while importing from CLU**

## Handling cancellations
### Cancellation Callback:
* Added on_cancel() which gets called if the client cancels the request. It sets the self.is_cancelled flag.
* Registered this callback with context.add_callback(on_cancel).

### Cancellation Checks:
* Periodically checked self.is_cancelled.is_set() after every major step (project creation, workspace import, training).
* Called context.abort(grpc.StatusCode.CANCELLED, "Training cancelled by client.") if cancellation is detected. This is how the server informs the client that the gRPC operation has been cancelled, and it stops further processing on the server side.

### Threading Event:
* Used threading.Event() to handle cancellation and ensure thread-safe checking of the cancellation status.


## Docker commands

### Build image
sudo docker build . -t clu-custom-connector:latest --no-cache

### Create, attach and run the commands manually
sudo docker run -e "CLU_KEY=$CLU_KEY" -e "CLU_ENDPOINT=$CLU_ENDPOINT" -e "HF_USERNAME=$HF_USERNAME" -e "HF_PASSWORD=$HF_PASSWORD" -d --name clu-custom-connector-3 -p 443:443 clu-custom-connector tail -f /dev/null

sudo docker exec -it clu-custom-connector-3 /bin/bash

poetry run python3 -m hf_integration.main ./credentials/my_org-mtls-credentials.json 0.0.0.0:443 clu "clu_endpoint::$CLU_ENDPOINT,clu_key::$CLU_KEY,delimiter::-,project_path::./,clu_la
nguage::en-us,clu_multilingual::True,clu_training_mode::standard,log_level::debug,max_batch_size::500"

### Run the commands while creating the container
sudo docker run -e "CLU_KEY=$CLU_KEY" -e "CLU_ENDPOINT=$CLU_ENDPOINT" -e "HF_USERNAME=$HF_USERNAME" -e "HF_PASSWORD=$HF_PASSWORD" -d --name clu-custom-connector-4 -p 443:443 clu-custom-connector poetry run python3 -m hf_integration.main ./credentials/my_org-mtls-credentials.json 0.0.0.0:443 clu "clu_endpoint::$CLU_ENDPOINT,clu_key::$CLU_KEY,delimiter::-,project_path::./,clu_language::en-us,clu_multilingual::True,clu_training_mode::standard,log_level::debug,max_batch_size::500"

### Free up the port 443
sudo kill -9 $(sudo lsof -t -i :443)


## Log handling