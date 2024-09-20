:mod:`{{module}}`.{{objname}}
{{ underline }}==============

.. currentmodule:: {{module}}

.. autoclass:: {{ objname }}

   {% block methods %}
    .. automethod:: __call__
    .. automethod:: create_iterator
    .. automethod:: load_default_package
    .. automethod:: load_model
   {% endblock %}

.. raw:: html

    <div class="clearer"></div>
