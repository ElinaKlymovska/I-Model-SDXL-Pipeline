"""
Dependency Injection Container
Provides centralized dependency management for the pipeline components.
"""

from typing import Dict, Any, TypeVar, Type, Callable, Optional
import logging
from functools import lru_cache

from .config import ConfigManager
from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)

T = TypeVar('T')


class Container:
    """Simple dependency injection container"""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._singletons: Dict[str, Any] = {}
        self._config_manager: Optional[ConfigManager] = None
    
    def register_singleton(self, interface: Type[T], implementation: Type[T], name: str = None) -> 'Container':
        """Register a singleton service"""
        service_name = name or interface.__name__
        self._factories[service_name] = lambda: implementation(self.get_config_manager())
        return self
    
    def register_factory(self, interface: Type[T], factory: Callable[[], T], name: str = None) -> 'Container':
        """Register a factory function"""
        service_name = name or interface.__name__
        self._factories[service_name] = factory
        return self
    
    def register_instance(self, interface: Type[T], instance: T, name: str = None) -> 'Container':
        """Register a specific instance"""
        service_name = name or interface.__name__
        self._services[service_name] = instance
        return self
    
    def register_config_manager(self, config_manager: ConfigManager) -> 'Container':
        """Register the configuration manager"""
        self._config_manager = config_manager
        self._services['ConfigManager'] = config_manager
        return self
    
    def get(self, interface: Type[T], name: str = None) -> T:
        """Get a service instance"""
        service_name = name or interface.__name__
        
        # Check if already instantiated
        if service_name in self._services:
            return self._services[service_name]
        
        # Check if it's a singleton
        if service_name in self._singletons:
            return self._singletons[service_name]
        
        # Check if we have a factory
        if service_name in self._factories:
            instance = self._factories[service_name]()
            
            # Store singletons
            if hasattr(instance, '__class__') and getattr(instance.__class__, '_singleton', False):
                self._singletons[service_name] = instance
            
            return instance
        
        raise ConfigurationError(f"Service '{service_name}' not registered in container")
    
    def get_config_manager(self) -> ConfigManager:
        """Get the configuration manager"""
        if self._config_manager is None:
            raise ConfigurationError("ConfigManager not registered in container")
        return self._config_manager
    
    def has(self, interface: Type[T], name: str = None) -> bool:
        """Check if a service is registered"""
        service_name = name or interface.__name__
        return (service_name in self._services or 
                service_name in self._factories or 
                service_name in self._singletons)
    
    def clear(self):
        """Clear all registrations (useful for testing)"""
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()
        self._config_manager = None


# Global container instance
_container: Optional[Container] = None


def get_container() -> Container:
    """Get the global container instance"""
    global _container
    if _container is None:
        _container = Container()
    return _container


def reset_container():
    """Reset the global container (useful for testing)"""
    global _container
    if _container:
        _container.clear()
    _container = None


@lru_cache(maxsize=None)
def create_default_container() -> Container:
    """Create a container with default service registrations"""
    from .config import get_config_manager
    
    container = Container()
    
    # Register configuration manager
    config_manager = get_config_manager()
    container.register_config_manager(config_manager)
    
    return container


class ServiceRegistry:
    """
    Registry for managing service lifecycle and dependencies.
    Provides a higher-level interface for common dependency injection patterns.
    """
    
    def __init__(self, container: Container = None):
        self.container = container or get_container()
        self._initialized_services = set()
    
    def initialize_service(self, service_type: Type[T], *args, **kwargs) -> T:
        """Initialize a service with proper dependency injection"""
        service_name = service_type.__name__
        
        if service_name in self._initialized_services:
            return self.container.get(service_type)
        
        try:
            # Check if service requires config manager
            if hasattr(service_type.__init__, '__annotations__'):
                annotations = service_type.__init__.__annotations__
                if 'config_manager' in annotations:
                    kwargs['config_manager'] = self.container.get_config_manager()
            
            # Create instance
            instance = service_type(*args, **kwargs)
            
            # Register instance
            self.container.register_instance(service_type, instance)
            self._initialized_services.add(service_name)
            
            logger.debug(f"Initialized service: {service_name}")
            return instance
            
        except Exception as e:
            logger.error(f"Failed to initialize service {service_name}: {e}")
            raise ConfigurationError(f"Service initialization failed: {service_name}") from e
    
    def get_or_create_service(self, service_type: Type[T], *args, **kwargs) -> T:
        """Get existing service or create new one"""
        if self.container.has(service_type):
            return self.container.get(service_type)
        return self.initialize_service(service_type, *args, **kwargs)
    
    def cleanup_services(self):
        """Cleanup all registered services"""
        for service_name in list(self._initialized_services):
            try:
                if self.container.has(type(None), service_name):
                    service = self.container.get(type(None), service_name)
                    if hasattr(service, 'cleanup'):
                        service.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up service {service_name}: {e}")
        
        self._initialized_services.clear()


def singleton(cls):
    """Decorator to mark a class as singleton"""
    cls._singleton = True
    return cls