import kfp
from google.cloud import aiplatform

from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp
from google_cloud_pipeline_components.v1.model import ModelUploadOp
from google_cloud_pipeline_components.v1.endpoint import ModelUndeployOp
from google_cloud_pipeline_components.v1.endpoint import ModelDeployOp
from google_cloud_pipeline_components.v1.endpoint import EndpointCreateOp
from google_cloud_pipeline_components.v1.model import ModelGetOp


project_id = "mlops2024-441121"
location = "us-central1"
pipeline_root="gs://mlops2024-pipeline/pipe-runs"

from kfp.v2.dsl import component, Input, Output, Artifact
from kfp.v2.dsl import component, OutputPath

# Component to undeploy all models from an endpoint
@component(
		packages_to_install=['google-cloud-aiplatform'],
		base_image='python:3.8',
)
def undeploy_all_models_from_endpoint(
		project_id: str,
		location: str,
		endpoint_resource: Input[Artifact],
):
		from google.cloud import aiplatform
		aiplatform.init(project=project_id, location=location)
		
		endpoint = aiplatform.Endpoint(endpoint_resource.uri.split("/")[-1])
		deployed_models = endpoint.list_models()
		
		for deployed_model in deployed_models:
				print(f"Undeploying model: {deployed_model.model}")
				endpoint.undeploy(
						deployed_model_id=deployed_model.id,
						sync=True,
				)


# Component to log the model resource output
@component
def log_model_upload_output(model_resource: Input[Artifact]):
		print("Model resource URI:", model_resource.uri)
		print("Model resource NAME:", model_resource.name)

# Custom component to upload a model to Vertex AI
@component(
		packages_to_install=['google-cloud-aiplatform'],
		base_image='python:3.8',
)
def upload_model_custom(
		project_id: str,
		location: str,
		display_name: str,
		artifact_uri: str,
		serving_container_image_uri: str,
		parent_model: str = '',
		model_resource: Output[Artifact] = None,
):
		from google.cloud import aiplatform
		aiplatform.init(project=project_id, location=location)
		model_resource = model_resource or Output[Artifact]()  
		model = aiplatform.Model.upload(
				display_name=display_name,
				artifact_uri=artifact_uri,
				serving_container_image_uri=serving_container_image_uri,
				parent_model=parent_model if parent_model else None,
		)
		model.wait()
		# Output the resource name for downstream components
		print("MODEL RESOURCE NAME: ", model.resource_name)
		print("VERSIONED MODEL RESOURCE NAME: ", model.versioned_resource_name)
		model_resource.uri = model.versioned_resource_name
		
		
		

# Define the pipeline
@kfp.dsl.pipeline(name="effort-training-deployment-pipeline", pipeline_root=pipeline_root)
def effort_pipeline():
		
		# Define the training job step - will be used for a scikit-learn model
		training_job = CustomTrainingJobOp(
				project=project_id,
				location=location,
				display_name="effort-training",
				worker_pool_specs=[
						{
								"machine_spec": {
										"machine_type": "n1-standard-4",
								},
								"replica_count": 1,
								"container_spec": {
										"image_uri": "us-central1-docker.pkg.dev/mlops2024-441121/effort/effort:latest",
								},
						}
				],
				base_output_directory="gs://effort-activity/model-output/",
		)
		existing_model_resource_name = f"projects/{project_id}/locations/{location}/models/8982047926255616000"
		# Model upload step
		model_upload_op = upload_model_custom(
		project_id=project_id,
		location=location,
		display_name="effort",
		artifact_uri='gs://effort-activity/model-output/',
		serving_container_image_uri="us-docker.pkg.dev/vertex-ai-restricted/prediction/tf_opt-cpu.2-13:latest",
		parent_model=existing_model_resource_name,  # Include if updating an existing model
		)	
		model_upload_op.after(training_job)
		# **Add Logging Step Here**
		log_model_op = log_model_upload_output(
				model_resource=model_upload_op.outputs['model_resource']
		)
		log_model_op.after(model_upload_op)
		# Retrieve the versioned model resource name
		get_model_version_op = ModelGetOp(
				project=project_id,
				model_name='8982047926255616000',
		)
		get_model_version_op.after(model_upload_op)
		
		create_endpoint_op = EndpointCreateOp(
				project=project_id,
				location=location,
				display_name = "effort-endpoint",
				labels={"test": "test"},
		)
		create_endpoint_op.after(get_model_version_op)
		
		undeploy_op = undeploy_all_models_from_endpoint(
				project_id=project_id,
				location=location,
				endpoint_resource=create_endpoint_op.outputs["endpoint"],
		)
		undeploy_op.after(create_endpoint_op)
		# LOG MODEL UPLOAD OUTPUT MODEL
		# Model deploy step
		print(create_endpoint_op.outputs)
		model_deploy_op = ModelDeployOp(
				model=get_model_version_op.outputs["model"],
				endpoint=create_endpoint_op.outputs["endpoint"],
				dedicated_resources_machine_type="n1-standard-4",
				dedicated_resources_min_replica_count=1,
		)
		model_deploy_op.after(undeploy_op)
	

		
		
		

		
from kfp.v2 import compiler

# Compile the pipeline
compiler.Compiler().compile(
		pipeline_func=effort_pipeline,
		package_path="effort_pipeline.json"
)