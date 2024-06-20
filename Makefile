# Include OCI Images support
-include .make/oci.mk

# Include k8s support
-include .make/k8s.mk

# Include Helm Chart support
-include .make/helm.mk

# Include Python support
-include .make/python.mk

# Include raw support
-include .make/raw.mk

# Include core make support
-include .make/base.mk

# Include your own private variables for custom deployment configuration
-include PrivateRules.mak

