from pydantic import BaseModel, Field, PrivateAttr
from typing import List, Dict, Type, Union, Tuple, Optional, Literal
from guardrails import Guard
from guardrails.validator_base import Validator
from guardrails.classes.validation_outcome import ValidationOutcome
from config.guardrails_config import GUARDRAIL_CONFIG, GUARDRAIL_POLICIES

ValidationStrategy = Literal["solo", "multiple", "all"]

class ValidationException(ValueError):
    """Exception raised when validation fails"""
    def __init__(self, message, results):
        super().__init__(message)
        self.results = results

class GuardValidationResult(BaseModel):
    """Structured validation result"""
    passed: bool
    message: Optional[str] = None
    details: Optional[Dict] = None
    raw_output: Optional[str] = None

class Guardrails(BaseModel):
    """Enhanced Guardrails implementation with better type safety and validation handling"""
    solo_guards: Dict[str, Guard] = Field(default_factory=dict)  # Using dict for O(1) lookup
    multiple_guards: Optional[Guard] = None

    # Private configurations
    _solo_guards_config: Dict[str, Tuple[Type[Validator], Dict]] = PrivateAttr(default_factory=dict)
    _multiple_guards_config: List[Union[Validator, Tuple[Type[Validator], Dict]]] = PrivateAttr(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    def with_guards_for(self, categories: List[str]) -> "Guardrails":
        """
        Adds guards based on predefined categories.

        Args:
            categories: A list of category names to add guards for.

        Returns:
            self for method chaining.
        """
        for category in categories:
            if category not in GUARDRAIL_CONFIG:
                print(f"Warning: Guardrail category '{category}' not found. Skipping.")
                continue

            for validator_cls, kwargs in GUARDRAIL_CONFIG[category]:
                # Create a unique name to prevent duplicates if categories overlap
                guard_name = f"{category}_{validator_cls.__name__}"
                if guard_name not in self.solo_guards:
                    self.add_solo_guard(validator_cls, name=guard_name, **kwargs)
        return self

    def with_policy(self, policy_name: str) -> "Guardrails":
        """
        Applies a predefined policy by adding all guards from the policy's categories.

        Args:
            policy_name: The name of the policy to apply ('strict', 'moderate', 'basic').

        Returns:
            self for method chaining.
        """
        if policy_name not in GUARDRAIL_POLICIES:
            raise ValueError(f"Policy '{policy_name}' not found. Available policies: {list(GUARDRAIL_POLICIES.keys())}")
        
        categories = GUARDRAIL_POLICIES[policy_name]
        return self.with_guards_for(categories)

    def add_solo_guard(
        self, 
        validator_cls: Type[Validator], 
        name: Optional[str] = None,
        **kwargs
    ) -> "Guardrails":
        """
        Add a single guard with its configuration
        
        Args:
            validator_cls: The validator class to use
            name: Optional custom name for the guard
            **kwargs: Configuration for the validator
        
        Returns:
            self for method chaining
        
        Raises:
            ValueError: If guard initialization fails
        """
        try:
            guard_name = name or validator_cls.__name__
            if guard_name in self.solo_guards:
                raise ValueError(f"Guard with name '{guard_name}' already exists")

            guard = Guard().use(validator_cls, **kwargs)
            self.solo_guards[guard_name] = guard
            self._solo_guards_config[guard_name] = (validator_cls, kwargs)
            
        except Exception as e:
            raise ValueError(f"Error initializing solo guard {validator_cls.__name__}: {str(e)}")
        
        return self

    def add_multiple_guards(
        self, 
        *validators: Union[Validator, Tuple[Type[Validator], Dict]]
    ) -> "Guardrails":
        """
        Add multiple guards to be used together
        
        Args:
            *validators: Either validator instances or tuples of (validator class, config)
        
        Returns:
            self for method chaining
        
        Raises:
            ValueError: If guards initialization fails
        """
        try:
            self.multiple_guards = Guard().use_many(*validators)
            self._multiple_guards_config = list(validators)
        except Exception as e:
            raise ValueError(f"Error initializing multiple guards: {str(e)}")
        
        return self

    def remove_guard(self, name: str) -> bool:
        """
        Remove a guard by name
        
        Returns:
            bool: True if guard was removed, False if not found
        """
        if name in self.solo_guards:
            del self.solo_guards[name]
            del self._solo_guards_config[name]
            return True
        return False

    def reset_guards(self) -> "Guardrails":
        """Reset all guards to initial state"""
        self.solo_guards.clear()
        self._solo_guards_config.clear()
        self.multiple_guards = None
        self._multiple_guards_config.clear()
        return self

    def get_guard(self, name: str) -> Optional[Guard]:
        """Get a guard by name"""
        return self.solo_guards.get(name)

    def get_guard_names(self) -> List[str]:
        """Get list of all guard names"""
        return list(self.solo_guards.keys())

    def _process_validation_outcome(
        self, 
        outcome: ValidationOutcome
    ) -> GuardValidationResult:
        """Convert ValidationOutcome to structured result"""
        # Extract validation summaries and convert to dict format
        details = None
        if hasattr(outcome, 'validation_summaries') and outcome.validation_summaries:
            details = {
                'summaries': [summary.to_dict() if hasattr(summary, 'to_dict') else str(summary) 
                             for summary in outcome.validation_summaries]
            }
        
        return GuardValidationResult(
            passed=outcome.validation_passed,
            message=outcome.error if not outcome.validation_passed else None,
            details=details,
            raw_output=outcome.raw_llm_output
        )

    def validate(
        self, 
        text: str, 
        strategy: ValidationStrategy = "all",
        raise_on_fail: bool = False
    ) -> Dict[str, Union[GuardValidationResult, Dict[str, GuardValidationResult]]]:
        """
        Validate text using specified strategy
        
        Args:
            text: Text to validate
            strategy: Validation strategy ('solo', 'multiple', or 'all')
            raise_on_fail: Whether to raise exception on validation failure
        
        Returns:
            Dictionary containing validation results
        
        Raises:
            ValueError: If validation fails and raise_on_fail is True
        """
        # Validate strategy parameter
        valid_strategies = ["solo", "multiple", "all"]
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy '{strategy}'. Must be one of: {valid_strategies}")
        
        results: Dict[str, Union[GuardValidationResult, Dict[str, GuardValidationResult]]] = {
            'solo_guards': {},
            'multiple_guards': None
        }

        validation_failed = False

        if strategy in ("solo", "all"):
            solo_results = {}
            for name, guard in self.solo_guards.items():
                try:
                    outcome = guard.validate(text)
                    result = self._process_validation_outcome(outcome)
                    if not result.passed:
                        validation_failed = True
                    solo_results[name] = result
                except Exception as e:
                    solo_results[name] = GuardValidationResult(
                        passed=False,
                        message=f"Validation failed: {str(e)}"
                    )
                    validation_failed = True
            results['solo_guards'] = solo_results

        if strategy in ("multiple", "all") and self.multiple_guards:
            try:
                outcome = self.multiple_guards.validate(text)
                result = self._process_validation_outcome(outcome)
                if not result.passed:
                    validation_failed = True
                results['multiple_guards'] = result
            except Exception as e:
                results['multiple_guards'] = GuardValidationResult(
                    passed=False,
                    message=f"Multiple guards validation failed: {str(e)}"
                )
                validation_failed = True

        if raise_on_fail and validation_failed:
            raise ValidationException("Validation failed. Check results for details.", results)

        return results